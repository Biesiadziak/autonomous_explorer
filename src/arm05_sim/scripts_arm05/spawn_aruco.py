#!/usr/bin/env python3

import argparse
import os
import subprocess
from pathlib import Path

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
import yaml
from typing import List, Optional, Tuple


class ArucoSpawner(Node):

    def __init__(
        self,
        aruco_nr: int,
        fallback_positions: List[Tuple[float, float]],
        min_clearance: float = 0.4,
        min_separation: float = 0.6,
        position_jitter: float = 0.1,
    ):
        super().__init__('aruco_spawner')

        self.aruco_nr = aruco_nr
        self.fallback_positions = list(fallback_positions)
        self.min_clearance = float(min_clearance)
        self.min_separation = float(min_separation)
        self.position_jitter = float(position_jitter)

        share_dir = Path(get_package_share_directory('arm05_sim'))
        self.aruco_models_dir = share_dir / 'models' / 'aruco_markers'
        self.model_filename = 'model.sdf'

        self.final_aruco_positions = []
        self.free_cells: List[Tuple[int, int]] = []
        self.map_resolution: Optional[float] = None
        self.map_origin: Tuple[float, float] = (0.0, 0.0)
        self.map_shape: Optional[Tuple[int, int]] = None
        self.clearance_cells: int = 0
        self.free_mask: Optional[np.ndarray] = None

        map_yaml_path = share_dir / 'map' / 'map.yaml'
        self._setup_map_sampling(map_yaml_path)

    def spawn_aruco_cubes(self):
        for i in range(self.aruco_nr):
            position = self._sample_position()
            if position is None:
                self.get_logger().error(
                    "Unable to find a valid spawn position for aruco_%d. Skipping.", i
                )
                continue

            x, y = position
            fpath = self.aruco_models_dir / str(i) / self.model_filename
            with fpath.open('r') as file:
                xml_str = file.read()
            name = f'aruco_{i}'

            cmd = [
                'ros2', 'run', 'ros_gz_sim', 'create',
                '-name', name,
                '-file', str(fpath),
                '-x', f'{x}',
                '-y', f'{y}',
                '-z', '0.0',
            ]
            # print("Command to spawn aruco:", ' '.join(cmd))  # Debug print statement
            self.get_logger().info("Command to spawn aruco: " + ' '.join(cmd))

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                self.get_logger().error(
                    f"Failed to spawn '{name}': {result.stderr.strip()}"
                )
                continue

            self.get_logger().info(result.stdout.strip())
            self.get_logger().info(
                "Spawned aruco_%d at (%.2f, %.2f)" % (i, x, y)
            )
            self.final_aruco_positions.append((x, y))

    def log_aruco_positions(self, log_path: Path | None = None):
        def log_file(self: 'MinimalClientAsync', filename: Path):
            with filename.open('w') as file:
                for i, pose in enumerate(self.final_aruco_positions):
                    file.write(f'Aruco {i}: ')
                    file.write(f'x: {pose[0]} ')
                    file.write(f'y: {pose[1]}\n')

        if log_path is None:
            log_path = Path(os.path.expanduser('~')) / '.ros' / 'arm05_sim' / 'logs'

        log_path.mkdir(parents=True, exist_ok=True)

        timestamp = self.get_clock().now().seconds_nanoseconds()[0]
        log_file(self, log_path / 'latest.txt')
        log_file(self, log_path / f'{timestamp}.txt')

    def _setup_map_sampling(self, map_yaml_path: Path) -> None:
        if not map_yaml_path.exists():
            self.get_logger().warning(
                f"Map file '{map_yaml_path}' not found; falling back to predefined Aruco positions."
            )
            return

        try:
            with map_yaml_path.open('r') as file:
                map_config = yaml.safe_load(file)
        except (yaml.YAMLError, OSError) as exc:
            self.get_logger().warning(
                f"Failed to parse map configuration '{map_yaml_path}': {exc}. "
                "Falling back to predefined positions."
            )
            return

        image_path = (map_yaml_path.parent / map_config['image']).resolve()
        try:
            map_image, max_value = self._read_pgm(image_path)
        except Exception as exc:  # broad, but we log and fall back
            self.get_logger().warning(
                f"Unable to load map image '{image_path}': {exc}. "
                "Falling back to predefined positions."
            )
            return

        resolution = float(map_config.get('resolution', 0.05))
        origin = map_config.get('origin', [0.0, 0.0, 0.0])
        origin_x, origin_y = float(origin[0]), float(origin[1])

        mode = map_config.get('mode', 'trinary')
        free_mask = self._build_free_mask(map_image, max_value, mode, map_config)
        if not free_mask.any():
            self.get_logger().warning(
                f"Computed free space mask from '{image_path}' is empty; using predefined positions."
            )
            return

        self.map_resolution = resolution
        self.map_origin = (origin_x, origin_y)
        self.map_shape = free_mask.shape
        self.free_mask = free_mask

        self.clearance_cells = int(np.ceil(self.min_clearance / resolution))

        candidates = np.argwhere(free_mask)
        if candidates.size == 0:
            self.get_logger().warning(
                "No candidate free cells remained after applying clearance filtering."
            )
            return

        self.free_cells = [tuple(idx) for idx in candidates.tolist()]
        self.get_logger().info(
            f"Loaded {len(self.free_cells)} candidate positions from map for random Aruco placement."
        )

    def _read_pgm(self, image_path: Path) -> Tuple[np.ndarray, int]:
        with image_path.open('rb') as file:
            header = file.readline().strip()
            if header != b'P5':
                raise ValueError(f"Unsupported PGM format '{header.decode()}', expected 'P5'.")

            line = file.readline()
            while line.startswith(b'#'):
                line = file.readline()
            width, height = map(int, line.strip().split())

            max_value_line = file.readline().strip()
            max_value = int(max_value_line)
            if max_value > 255:
                raise ValueError("Only 8-bit PGM images are supported.")

            data = file.read(width * height)
            if len(data) != width * height:
                raise ValueError("Incomplete PGM image data.")

            image = np.frombuffer(data, dtype=np.uint8).reshape((height, width))

        return image, max_value

    def _build_free_mask(
        self,
        image: np.ndarray,
        max_value: int,
        mode: str,
        map_config: dict,
    ) -> np.ndarray:
        # In trinary mode, free cells are near max_value (typically 254)
        if mode == 'trinary':
            free_mask = image >= max_value - 1
            # Explicitly remove unknown cells (usually value 205)
            free_mask &= image > int(max_value * 0.8)
            return free_mask

        # Generic fallback based on free threshold
        free_thresh = float(map_config.get('free_thresh', 0.25))
        threshold_value = int(round((1.0 - free_thresh) * max_value))
        return image >= threshold_value

    def _sample_position(self) -> Optional[Tuple[float, float]]:
        # Attempt to sample from map-derived free cells
        if self.free_cells and self.map_resolution and self.map_shape:
            attempts = 0
            max_attempts = min(len(self.free_cells), 500)
            while attempts < max_attempts and self.free_cells:
                attempts += 1
                idx = np.random.randint(0, len(self.free_cells))
                row, col = self.free_cells[idx]
                if not self._cell_has_clearance(row, col):
                    self.free_cells.pop(idx)
                    continue

                x, y = self._grid_to_world(row, col)
                x += np.random.uniform(-self.position_jitter, self.position_jitter)
                y += np.random.uniform(-self.position_jitter, self.position_jitter)

                if not self._is_separated(x, y):
                    continue

                self.free_cells.pop(idx)
                return x, y

            self.get_logger().warning(
                "Ran out of sampled map positions with sufficient clearance; "
                "falling back to predefined positions."
            )

        # Fallback to predefined anchors with jitter
        if not self.fallback_positions:
            return None

        anchor_idx = np.random.randint(0, len(self.fallback_positions))
        base_x, base_y = self.fallback_positions.pop(anchor_idx)
        x = base_x + np.random.uniform(-self.position_jitter, self.position_jitter)
        y = base_y + np.random.uniform(-self.position_jitter, self.position_jitter)

        if not self._is_separated(x, y):
            return self._sample_position()

        return x, y

    def _grid_to_world(self, row: int, col: int) -> Tuple[float, float]:
        assert self.map_resolution is not None and self.map_shape is not None
        height, _ = self.map_shape
        origin_x, origin_y = self.map_origin
        x = origin_x + (col + 0.5) * self.map_resolution
        y = origin_y + (height - row - 0.5) * self.map_resolution
        return x, y

    def _cell_has_clearance(self, row: int, col: int) -> bool:
        if self.free_mask is None:
            return False

        if self.clearance_cells <= 0:
            return bool(self.free_mask[row, col])

        r_min = max(row - self.clearance_cells, 0)
        r_max = min(row + self.clearance_cells + 1, self.free_mask.shape[0])
        c_min = max(col - self.clearance_cells, 0)
        c_max = min(col + self.clearance_cells + 1, self.free_mask.shape[1])
        patch = self.free_mask[r_min:r_max, c_min:c_max]
        if patch.size == 0:
            return False

        return np.all(patch)

    def _is_separated(self, x: float, y: float) -> bool:
        if not self.final_aruco_positions:
            return True
        for px, py in self.final_aruco_positions:
            if np.hypot(x - px, y - py) < self.min_separation:
                return False
        return True

def main(args=None):
    rclpy.init(args=args)
    aruco_positions = [
        [-4.93, -0.05],
        [3.9, 3.6],
        [7.95, 3.18],
        [11.6, 3.6],
        [16.2, 2.0],
        [16.2, -1.7],
        [12.2, -5.5],
        [4.1, -4.3],
        [11.1, -1.3]
    ]

    parser = argparse.ArgumentParser(
        prog="ArucoSpawner",
        description="This script spawns aruco cubes in the simulated gazebo world",
    )

    parser.add_argument(
        '-s',
        '--seed',
        type=int,
        help=(
            "The random seed for numpy. It can be used to obtain constant positions "
            "of the cubes."
        ),
    )
    args, _ = parser.parse_known_args()
    
    if args.seed is not None:
        np.random.seed(args.seed)

    spawner = ArucoSpawner(
        aruco_nr=5,
        fallback_positions=aruco_positions,
    )

    spawner.get_logger().info("Seed: %s" % args.seed)

    spawner.spawn_aruco_cubes()
    spawner.log_aruco_positions()

    spawner.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
