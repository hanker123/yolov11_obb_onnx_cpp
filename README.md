# yolov11_obb_onnx_cpp
Implementing Rotated Object Detection Inference with YOLOv11 OBB, C++, and ONNX for Cross-Platform CPU Deployment on Android and Linux

# Introduce
- Performing inference with YOLOv11 OBB ONNX on CPU using C++, with inference code implemented in C++ and leveraging frameworks such as OpenCV and ONNX, enabling cross-platform deployment on both Linux and Android platforms.
- This version implements single-object rotation detection and multi-object rotation detection. Code modifications are required manually, or updates will be made in future versions.

# Start
## 1、linux：
`g++ -o test yolo_obb_onnx.cpp -I include/ -L lib/ -lonnxruntime pkg-config --cflags --libs opencv`  
`./test`
## 2、android：
- On the Android platform, the main output is a .so (shared object) library file, which can be called by other Android applications.  
`bash android_compile.sh`
# Result
<img width="800" height="800" alt="image" src="https://github.com/user-attachments/assets/9824749d-8fe4-4382-802f-6b94674772b9" />
