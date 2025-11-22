# 🤖 Autonomous Explorer

[Watch the demo video](https://youtu.be/zZm_Slx6ymI)

---

## 🎯 Project Goal

Develop an autonomous system that **explores a known environment and detects Aruco markers**.

---

## 🛠 Selected Solution

- **Boustrophedon Algorithm** – deterministic full coverage of the area.  
- **OpenCV + ArUco** – marker detection and position estimation.  
- **ROS2 + Nav2** – navigation, path planning, spatial transformations.  
- **Python 3** – system integration and image processing.

---

## 🖼 How the System Works

1. The robot follows a planned trajectory.  
2. Camera captures images → converted to grayscale + thresholding.  
3. ArUco markers detected → positions recorded in the global frame.  
4. Multi-threading in ROS2 allows simultaneous navigation and image processing.  

---

## ⚠️ Potential Issues

- Slower movement and stopping between points 🚶‍♂️  
- Getting stuck near obstacles 🧱  
- Missing markers when camera angle is bad 📷  

**Solutions:** monitor point completion time, improve obstacle maps, stabilize camera images.

---

## ✅ Conclusions

- The system works and successfully locates markers.  
- Boustrophedon ensures deterministic area coverage.  
- Provides a solid foundation for further experiments in autonomous exploration.
