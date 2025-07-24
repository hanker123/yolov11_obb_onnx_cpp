g++ -o test yolo11_obb_onnx.cpp -I core/include/ -L lib/ -lonnxruntime `pkg-config --cflags --libs opencv`
