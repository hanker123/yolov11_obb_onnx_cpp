//Author Hanchaoyang 
//version android so
#ifndef FAN_MODEL_H_
#define FAN_MODEL_H_


#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <string>
#include <utility>
#include <fstream>

#include <opencv2/opencv.hpp>
#include <onnx/onnxruntime_cxx_api.h>
#include <jni.h>


using namespace std;
using namespace cv;


struct Convariance_Matrix
{
    vector<float> cov_xx;
    vector<float> cov_yy;
    vector<float> cov_xy;
};


struct Obb_Res
{
    vector<float> point;
    float label;
    float score;
};


//函数声明

Convariance_Matrix get_convariance_matrix(vector<vector<float>> boxes);

Obb_Res xywhr2xyxyxyxy(vector<float> obb);

vector<int> probiou(vector<vector<float>>obb1, vector<vector<float>>obb2, float eps);

vector<Obb_Res> model_predict(const string model_path, const string img_path, int targetWidth, int targetHeight, float* output_data, vector<int64_t>& outputDims, Mat& dest_image);

#endif
