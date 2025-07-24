// Author Hanchaoyang
// v1.0 android

#include "model.h"


Convariance_Matrix get_convariance_matrix(vector<vector<float>> boxes)
{
    Convariance_Matrix c_matrix;

    

    for (int i = 0; i < boxes.size(); i++)
    {
        vector<float> box = boxes[i];
        float a = pow(box[2], 2) / 12.0;
        float b = pow(box[3], 2) / 12.0;
        float c = box[4];

        float cos_c = cos(c);
        float sin_c = sin(c);

        float cov_xx_i = a * pow(cos_c, 2) + b * pow(sin_c, 2);
        float cov_yy_i = a * pow(sin_c, 2) + b * pow(cos_c, 2);
        float cov_xy_i = a * cos_c * sin_c - b * sin_c * cos_c;

        c_matrix.cov_xx.push_back(cov_xx_i);
        c_matrix.cov_yy.push_back(cov_yy_i);
        c_matrix.cov_xy.push_back(cov_xy_i);

    }

    return c_matrix;

}


Obb_Res xywhr2xyxyxyxy(vector<float> obb)
{
    /*
    obb: (x, y, w, h, angle)
    */
        
    
    Obb_Res point_info;
    float x = obb[0];
    float y = obb[1];
    float w = obb[2];
    float h = obb[3];
    float angle = obb[6];

    float cos_value = cos(angle);
    float sin_value = sin(angle);
    float vec1_x = w / 2 * cos_value;
    float vec1_y = w / 2 * sin_value;
    float vec2_x = -h / 2 * sin_value;
    float vec2_y = h / 2 * cos_value;

    vector<float> point;

    point.push_back(x + vec1_x + vec2_x);
    point.push_back(y + vec1_y + vec2_y);

    point.push_back(x + vec1_x - vec2_x);
    point.push_back(y + vec1_y - vec2_y);

    point.push_back(x - vec1_x - vec2_x);
    point.push_back(y - vec1_y - vec2_y);

    point.push_back(x - vec1_x + vec2_x);
    point.push_back(y - vec1_y + vec2_y);

    point_info.point = point;
    point_info.label = obb[5];
    point_info.score = obb[4];


    return point_info;

}



vector<int> probiou(vector<vector<float>>obb1, vector<vector<float>>obb2, float eps)
{
    Convariance_Matrix c_matrix1 = get_convariance_matrix(obb1);
    Convariance_Matrix c_matrix2 = get_convariance_matrix(obb2);

    

    vector<vector<float>> iou_list;
    
    float iou_threa = 0.7;

    for (int i = 0; i < obb1.size(); i++)
    {
        vector<float> iou;

        for (int j = 0; j < obb2.size(); j++)
        {
            vector<float> obb_1 = obb1[i];
            vector<float> obb_2 = obb2[j];

            float a1 = c_matrix1.cov_xx[i];
            float a2 = c_matrix2.cov_xx[j];

            float b1 = c_matrix1.cov_yy[i];
            float b2 = c_matrix2.cov_yy[j];

            float c1 = c_matrix1.cov_xy[i];
            float c2 = c_matrix2.cov_xy[j];

            float x1 = obb_1[0];
            float y1 = obb_1[1];

            float x2 = obb_2[0];
            float y2 = obb_2[1];

            float t1 = (((a1 + a2) * (pow(y1-y2, 2)) + (b1 + b2) * (pow(x1-x2,2))) / ((a1 + a2) * (b1 + b2) - (pow(c1+c2,2))+ eps)) * 0.25;

            float t2 = (((c1 + c2) * (x2 - x1) * (y1 - y2))/ ((a1 + a2) * (b1 + b2) - (pow(c1 + c2, 2)) + eps)) * 0.5;

            float d = (a1 * b1 - pow(c1, 2));

            if (d <=0)
            {
                d = 0;
            }

            float v = (a2 * b2 - pow(c2, 2));

            if (v <=0)
            {
                v = 0;
            }

            float t3 = (log(((a1 + a2) * (b1 + b2) - (pow(c1 + c2, 2)))/(4 * sqrt(d * v) + eps) + eps) * 0.5);

            float bd = t1 + t2 + t3;

            if (bd <= eps)
            {
                bd = eps;
            }
            if (bd >= 100.0)
            {
                bd = 100.0;
            }

            float hd = sqrt(1.0 - exp(-bd) + eps);

            float res = 1 - hd;

            iou.push_back(res);

        }
        iou_list.push_back(iou);
    }

    vector<int> iou_index;
    for (int i = 0; i < iou_list.size(); i++)
    {
        for (int j = i; j < iou_list.size(); j++)
        {
            // vector<float> data = iou_list[i];
            // data[j] = 0;
            iou_list[i][j] = 0.0;
        }
    }

    for (int i = 0; i < iou_list.size(); i++)
    {
        vector<float> inner_iou = iou_list[i];

        auto max_iter = max_element(inner_iou.begin(), inner_iou.end());
        float max_iou = 0.0;

        if(max_iter != inner_iou.end()) 
        {
            // 解引用迭代器以获取最大值
            max_iou = *max_iter;
        }

        if (max_iou < iou_threa)
        {
            iou_index.push_back(i);
        }

    }

    return iou_index;

}

vector<Obb_Res> model_predict(const string model_path, const string img_path, const string save_img_path, int targetWidth, int targetHeight, float* output_data, vector<int64_t>& outputDims, Mat& dest_image)
{

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "YOLOv11_OBB");
    Ort::SessionOptions sessionOptions;
    sessionOptions.SetIntraOpNumThreads(1);
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    //模型初始化
    Ort::Session model_session(env, ORT_TSTR(model_path.c_str()), sessionOptions);

    

    // 获取输入输出信息
    Ort::AllocatorWithDefaultOptions allocator;
    size_t numInputNodes = model_session.GetInputCount();
    size_t numOutputNodes = model_session.GetOutputCount();

    std::cout << "Number of inputs: " << numInputNodes << std::endl;
    std::cout << "Number of outputs: " << numOutputNodes << std::endl;

    // 假设输入和输出只有一个节点
    auto inputNamePtr = model_session.GetInputNameAllocated(0, allocator);
    auto outputNamePtr = model_session.GetOutputNameAllocated(0, allocator);

    //获取输入，输出节点name
    const char* inputName = inputNamePtr.get();
    const char* outputName = outputNamePtr.get();

    // 获取输入形状
    Ort::TypeInfo inputTypeInfo = model_session.GetInputTypeInfo(0);
    auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
    vector<int64_t> inputDims = inputTensorInfo.GetShape();
    cout << "Input dimensions: ";
    for (auto dim : inputDims) std::cout << dim << " ";
    std::cout << std::endl;


    // 2. 加载图像并预处理
    cv::Mat s_image = cv::imread(img_path);
    if (s_image.empty()) {
        std::cerr << "Error: Could not load image!" << std::endl;
        // return -1;
    }
    //图像预处理过程
    // 计算需要填充的边界大小
    int scale = 0;
    if (targetHeight < s_image.rows || targetHeight < s_image.cols)
    {
        scale = max(s_image.rows, s_image.cols);
    }
    else
    {
        scale = targetHeight;
    }
    int top = (scale - s_image.rows) / 2;    // 上边界
    int bottom = scale - s_image.rows - top; // 下边界
    int left = (scale - s_image.cols) / 2;                                         // 左边界
    int right = scale - s_image.cols - left;

    //构建4000*4000图像
    cv::Mat image;
    cv::copyMakeBorder(s_image, image, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(114,114,114));

    cv::resize(image, image, cv::Size(inputDims[3], inputDims[2])); // 调整到模型输入大小
    // dest image为输出图像
    dest_image = image.clone();
    Mat draw_image = image.clone();
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB); // 转换为 RGB
    image.convertTo(image, CV_32F, 1.0 / 255.0); // 归一化

    std::vector<float> inputTensorValues(image.rows * image.cols * image.channels());
    cv::Mat channels[3];
    cv::split(image, channels);
    for (int c = 0; c < 3; ++c) {
        std::memcpy(inputTensorValues.data() + c * image.rows * image.cols,
                    channels[c].data, sizeof(float) * image.rows * image.cols);
    }

    // 3. 运行推理
    std::vector<const char*> inputNames = {inputName};
    std::vector<const char*> outputNames = {outputName};

    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(memoryInfo, inputTensorValues.data(),
                                                                 inputTensorValues.size(), inputDims.data(), inputDims.size());

    auto outputTensors = model_session.Run(Ort::RunOptions{nullptr}, inputNames.data(), &inputTensor, 1, outputNames.data(), 1);

    // 获取输出数据 1 * 7 * 13125
    output_data = outputTensors[0].GetTensorMutableData<float>();

    std::vector<int64_t> outputDimss = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();
    outputDims = outputDimss;
    std::cout << "Output dimensions: ";
    for (auto dim : outputDimss) std::cout << dim << " ";
    std::cout << std::endl;

    model_session.release();
    sessionOptions.release();
    // allocator.release();

    float conf_thread = 0.7;

    vector<vector<float>> total_data = {};

    for (int i = 0; i < outputDimss[2]; i++)
    {
        
        float x = output_data[0 * outputDimss[2] + i];
        float y = output_data[1 * outputDimss[2] + i];
        float w = output_data[2 * outputDimss[2] + i];
        float h = output_data[3 * outputDimss[2] + i];
        float score1 = output_data[4 * outputDimss[2] + i];
        float score2 = output_data[5 * outputDimss[2] + i];
        float angle = output_data[6 * outputDimss[2] + i];

        // cout << "x: " << x << " y: " << y << " w: " << w << " h: " << h << " score1: " << score1 << " score2: " << score2 << " angle: "  << angle << endl;

        float max_score = max(score1, score2);
        
        //根据置信度进行筛选
        if (max_score >= conf_thread)
        {
            vector<float> data = {};
            float j = 0.0;
            if(max_score == score2)
            {
                j = 1.0;
            }

            data.push_back(x);
            data.push_back(y);
            data.push_back(w);
            data.push_back(h);
            data.push_back(max_score);
            data.push_back(j);
            data.push_back(angle);
            total_data.push_back(data);
        }

        
    }

    int n = total_data.size();
    cout << "n: " << n << endl;

    int max_nms = 30000;

    //先进行排序
    sort(total_data.begin(), total_data.end(), [](const std::vector<float>& a, const std::vector<float
        >& b) {
            return a[4] > b[4]; // 按降序排序
        });
    
    // cout << total_data.size() << endl;
    
    vector<vector<float>> f_data;
    if(n > max_nms) 
    {
        // 截取前 max_nms 个子向量
        if(max_nms > total_data.size()) {
            max_nms = total_data.size(); // 确保不会越界
        }
        f_data.assign(total_data.begin(), total_data.begin() + max_nms);
    } 
    else
    {
        // 直接复制 total_data
        f_data = total_data;
    }

    float agnostic = 0.0;
    float max_wh = 7680.0;

    vector<vector<float>> new_f_data;
    vector<float> scores;
    
    for (int i = 0; i < f_data.size(); i++)
    {
        vector<float> data = f_data[i];
        vector<float> new_inner_data;

       
        float add = 0.0;
        if(agnostic == 1.0)
        {
            add = 0.0 * data[5];
        }
        else
        {
            add = max_wh * data[5];
        }

        new_inner_data.push_back(data[0] + add);
        new_inner_data.push_back(data[1] + add);
        new_inner_data.push_back(data[2]);
        new_inner_data.push_back(data[3]);
        new_inner_data.push_back(data[6]);

        scores.push_back(data[4]);
        new_f_data.push_back(new_inner_data);
        
    }
    
    float eps = 1e-7;
    vector<int> index = probiou(new_f_data, new_f_data, eps);


    //获得最终框obb
    vector<Obb_Res> final_point;
    for (int j = 0; j < index.size(); j++)
    {
        cout << "index: " << index[j] << endl;
        vector<float> data = f_data[index[j]];

        Obb_Res point_info = xywhr2xyxyxyxy(data);

        final_point.push_back(point_info);


        // for (int i = 0; i < data.size(); i++)
        // {
        //     cout << data[i] << " ";
        // }
        cout << endl;
    }

    // 画图，进行可视化
     
    for (int i = 0; i < final_point.size(); i++)
    {
        vector<cv::Point> draw_point;
        Obb_Res bbox = final_point[i];
        cv::Point p1 = {static_cast<int>(bbox.point[0]), static_cast<int>(bbox.point[1])};
        cv::Point p2 = {static_cast<int>(bbox.point[2]), static_cast<int>(bbox.point[3])};
        cv::Point p3 = {static_cast<int>(bbox.point[4]), static_cast<int>(bbox.point[5])};
        cv::Point p4 = {static_cast<int>(bbox.point[6]), static_cast<int>(bbox.point[7])};
        draw_point.push_back(p1);
        draw_point.push_back(p2);
        draw_point.push_back(p3);
        draw_point.push_back(p4);

        const cv::Point* ppt = &draw_point[0]; // 获取点数组的指针
        int npt = static_cast<int>(draw_point.size()); // 顶点数
        cv::polylines(draw_image, &ppt, &npt, 1, true, cv::Scalar(0, 0, 255), 2); // 红色线条

    }
    imwrite(save_img_path, draw_image);
    // imwrite("f_1.jpg", dest_image);

    return final_point;
}


int run_main(string& model_path, string& img_path, string* save_img_path)
{
    //  model_session;

    // const string model_path = "best_obb_new.onnx";
    vector<int64_t> outpuDims;
    Mat dest_image;

    // Ort::Session model_session = model_init(model_path, inputName, outputName, inputDims);

    // const string img_path = "C.JPG";
    int targetWidth = 4000; 
    int targetHeight = 4000;
    float* out_data;

    vector<Obb_Res> flag = model_predict(model_path, img_path, save_img_path
                            targetWidth, 
                            targetHeight, 
                            out_data,
                            outpuDims,
                            dest_image);
    
    // int flag_1 = obb_process(out_data, outpuDims, 0.8);


    return 0;
}

// Java_com_glkj_windpatrol_FanNativeLib_00024Companion_run根据android项目中的名称确定
extern "C" JNIEXPORT jdoubleArray JNICALL
Java_com_glkj_windpatrol_FanNativeLib_00024Companion_run(JNIEnv* env, 
                                                        jobject /* this */, 
                                                        jstring model_path, 
                                                        jstring img_path,
                                                        jstring save_img_path,
                                                        ) {
  

    // 将 jstring 转换为 std::string
    const char* modelPath = env->GetStringUTFChars(model_path, nullptr);
    const char* imgPath = env->GetStringUTFChars(img_path, nullptr);
    const char* saveimgPath = env->GetStringUTFChars(save_img_path, nullptr);
   




    if (modelPath == nullptr || BimgPath == nullptr || CimgPath == nullptr || saveimgPath == nullptr || templatePath == nullptr || waylinesPath == nullptr
    ) {
            // 处理内存分配失败的情况
            if(modelPath) env->ReleaseStringUTFChars(model_path, modelPath);
            if(imgPath) env->ReleaseStringUTFChars(img_path, imgPath);
            if(saveimgPath) env->ReleaseStringUTFChars(save_img_path, saveimgPath);
           
            return nullptr;
    }

    string modelPathStr(modelPath);
    string imgPathStr(imgPath);
    string SaveImgPathStr(saveimgPath);

    vector<double> res1 = run_main(modelPathStr, imgPathStr, SaveImgPathStr);

    // 释放资源
    env->ReleaseStringUTFChars(model_path, modelPath);
    env->ReleaseStringUTFChars(img_path, imgPath);
    env->ReleaseStringUTFChars(save_img_path, saveimgPath);
   

    jdoubleArray array = env->NewDoubleArray(res1.size());
    if (array == nullptr) {
        return nullptr; // 内存分配失败
    }

    env->SetDoubleArrayRegion(array, 0, res1.size(), res1.data());

    return array;
    // return run();
}

