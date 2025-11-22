#!/usr/bin/env python3
import rclpy
from rclpy.action import ActionServer
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from rclpy.executors import MultiThreadedExecutor

from explorer_action.action import Explorer
from nav_msgs.msg import OccupancyGrid
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult

import numpy as np
from geometry_msgs.msg import PoseStamped, TransformStamped, Point
import math
import yaml

from cv_bridge import CvBridge
import cv2
import tf_transformations

from tf2_ros import TransformException, TransformBroadcaster
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from time import time, sleep

class ExplorerActionServer(Node):

    def __init__(self):
        super().__init__('explorer_action_server')
        self._action_server = ActionServer(
            self,
            Explorer,
            'explorer',
            self.execute_callback)
        
        self.map = None

        map_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history = HistoryPolicy.KEEP_LAST,
            depth = 1
        )

        self.map_subscriber = self.create_subscription(
            OccupancyGrid,
            "/map",
            self.map_callback,
            map_qos
        )

        self.image_subscriber = self.create_subscription(
            Image,
            "/camera/image_raw",
            self.image_callback,
            10
        )

        self.camera_info_subscriber = self.create_subscription(
            CameraInfo,
            "/camera/camera_info",
            self.camera_info_callback,
            10
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.pose_timeout = self.declare_parameter('waypoint_timeout_sec', 45.0).value

        self.cameraMatrix = None
        self.distCoeffs = None

        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
        self.parameters = cv2.aruco.DetectorParameters_create()

        self.cv_bridge = CvBridge()

        self.tf_broadcaster = TransformBroadcaster(self)

        self.points = {}

        self.get_logger().info('Explorer Action Server is running...')

    def map_callback(self, msg):
        self.map = msg

    def camera_info_callback(self, msg):

        self.cameraMatrix = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        self.distCoeffs = np.array(msg.d, dtype=np.float64)

    def image_callback(self, msg):
        image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        _, tresh = cv2.threshold(gray, 70 ,255 ,cv2.THRESH_BINARY)

        corners, ids, _ = cv2.aruco.detectMarkers(tresh, dictionary=self.dictionary, parameters=self.parameters)
        
        if self.distCoeffs is not None and ids is not None:
            self.get_logger().info(f'Detected: {ids}')
            for corner, marker_id in zip(corners, ids.flatten()):
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corner, 0.09, self.cameraMatrix, self.distCoeffs)

                t = TransformStamped()
                point = Point()

                t.header.stamp = self.get_clock().now().to_msg()
                t.header.frame_id = "camera_rgb_optical_frame"
                t.child_frame_id = f"aruco_{marker_id}"

                t.transform.translation.x = tvecs[0][0][0]
                t.transform.translation.y= tvecs[0][0][1]
                t.transform.translation.z = tvecs[0][0][2]

                try:
                    map_to_camera_ts = self.tf_buffer.lookup_transform(
                        "map",
                        "camera_rgb_optical_frame",
                        rclpy.time.Time())
                except TransformException as ex:
                    self.get_logger().info(
                        f'Could not transform map to camera_rgb_optical_frame: {ex}')
                    return
                
                # Build 4x4 matrix for map <- camera transform
                trans = map_to_camera_ts.transform.translation
                rot = map_to_camera_ts.transform.rotation
                map_T_camera = tf_transformations.quaternion_matrix([rot.x, rot.y, rot.z, rot.w])
                map_T_camera[0:3, 3] = [trans.x, trans.y, trans.z]

                # Build 4x4 matrix for camera <- aruco (from rvecs/tvecs)
                camera_T_aruco = np.eye(4)
                camera_T_aruco[0:3, 0:3] = cv2.Rodrigues(rvecs[0])[0]
                camera_T_aruco[0:3, 3] = [float(tvecs[0][0][0]), float(tvecs[0][0][1]), float(tvecs[0][0][2])]

                # Compute map <- aruco = (map <- camera) * (camera <- aruco)
                T = map_T_camera @ camera_T_aruco
                
                point.x = float(T[0, 3])
                point.y = float(T[1, 3])
                point.z = float(T[2, 3])

                if marker_id not in self.points:
                    self.points[marker_id] = point
                    
                rotation_matrix = np.eye(4)
                rotation_matrix[0:3, 0:3] = cv2.Rodrigues(rvecs[0])[0]
                q = tf_transformations.quaternion_from_matrix(rotation_matrix)

                t.transform.rotation.x = q[0]
                t.transform.rotation.y = q[1]
                t.transform.rotation.z = q[2]
                t.transform.rotation.w = q[3]

                self.tf_broadcaster.sendTransform(t)

                self.get_logger().info(f"{self.points}")

    def execute_callback(self, goal_handle):
        self.get_logger().info('Executing goal...')

        feedback_msg = Explorer.Feedback()
        
        nav = BasicNavigator()
        nav.waitUntilNav2Active()

        with open('src/explorer_action/pose_output.yaml', 'r') as file:
            data = yaml.safe_load(file)

        poses = []
        for key in sorted(k for k in data if k != 'updatetime'):
            pose_data = data[key]
            pose = PoseStamped()
            pose.header.frame_id = 'map'
            pose.pose.position.x = pose_data['position']['x']
            pose.pose.position.y = pose_data['position']['y']
            pose.pose.position.z = pose_data['position']['z']
            pose.pose.orientation.x = pose_data['orientation']['x']
            pose.pose.orientation.y = pose_data['orientation']['y']
            pose.pose.orientation.z = pose_data['orientation']['z']
            pose.pose.orientation.w = pose_data['orientation']['w']
            poses.append(pose)

        if not poses:
            self.get_logger().error('No waypoints found in YAML.')
            goal_handle.abort()
            return Explorer.Result()

        start_pose = poses[0]
        start_pose.header.stamp = self.get_clock().now().to_msg()
        # nav.setInitialPose(start_pose)
        # nav.waitUntilNav2Active()

        for idx, pose in enumerate(poses):
            result = Explorer.Result()

            pose.header.stamp = self.get_clock().now().to_msg()
            nav.goToPose(pose)

            start_time = time()
            timed_out = False

            while not nav.isTaskComplete():
                if goal_handle.is_cancel_requested:
                    self.get_logger().info('Cancel requested; stopping navigation.')
                    nav.cancelTask()
                    goal_handle.canceled()

                    return result

                if time() - start_time > self.pose_timeout:
                    timed_out = True
                    self.get_logger().warn(f'Waypoint {idx} exceeded {self.pose_timeout}s; skipping.')
                    nav.cancelTask()
                    while not nav.isTaskComplete():
                        sleep(0.5)
                    break
                
                positions = list(self.points.values())
                result.aruco_pos = positions
                feedback_msg.num_found = float(len(positions))
                goal_handle.publish_feedback(feedback_msg)

                sleep(0.5)

            if timed_out:
                continue

            result_state = nav.getResult()
            if result_state != TaskResult.SUCCEEDED:
                self.get_logger().warn(f'Navigator returned {result_state}; skipping waypoint {idx}.')
                continue

        goal_handle.succeed()
        result = Explorer.Result()
        positions = list(self.points.values())
        result.aruco_pos = positions

        return result
    
def main(args=None):
    rclpy.init(args=args)

    explorer_action_server = ExplorerActionServer()

    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(explorer_action_server)

    try:
        print("")
        explorer_action_server.get_logger().info('End with CTRL-C')
        executor.spin()
    except KeyboardInterrupt:
        explorer_action_server.get_logger().info('KeyboardInterrupt, shutting down.\n')

    explorer_action_server.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()