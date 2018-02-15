#include "DetectionBasedTracker_jni.h"
#include <opencv2/core.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/face.hpp>
#include <string>
#include <vector>

#include <android/log.h>

#define LOG_TAG "FaceDetection/DetectionBasedTracker"
#define LOGD(...) ((void)__android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__))
#define LOGE(...) ((void)__android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__))

using namespace std;
using namespace cv;
using namespace cv::face;

vector<Mat> images;
vector<int> labels;
Ptr<EigenFaceRecognizer> globalEigenModel;
Ptr<LBPHFaceRecognizer> globalLBPHModel;
Ptr<FisherFaceRecognizer> globalFisherModel;

inline void vector_Rect_to_Mat(vector<Rect> &v_rect, Mat &mat) {
    mat = Mat(v_rect, true);
}

class CascadeDetectorAdapter : public DetectionBasedTracker::IDetector {
public:
    CascadeDetectorAdapter(cv::Ptr<cv::CascadeClassifier> detector) :
            IDetector(),
            Detector(detector) {
        LOGD("CascadeDetectorAdapter::Detect::Detect");
        CV_Assert(detector);
    }

    void detect(const cv::Mat &Image, std::vector<cv::Rect> &objects) {
        LOGD("CascadeDetectorAdapter::Detect: begin");
        LOGD("CascadeDetectorAdapter::Detect: scaleFactor=%.2f, minNeighbours=%d, minObjSize=(%dx%d), maxObjSize=(%dx%d)",
             scaleFactor, minNeighbours, minObjSize.width, minObjSize.height, maxObjSize.width,
             maxObjSize.height);
        Detector->detectMultiScale(Image, objects, scaleFactor, minNeighbours, 0, minObjSize,
                                   maxObjSize);
        LOGD("CascadeDetectorAdapter::Detect: end");
    }

    virtual ~CascadeDetectorAdapter() {
        LOGD("CascadeDetectorAdapter::Detect::~Detect");
    }

private:
    CascadeDetectorAdapter();

    cv::Ptr<cv::CascadeClassifier> Detector;
};

struct DetectorAgregator {
    cv::Ptr<CascadeDetectorAdapter> mainDetector;
    cv::Ptr<CascadeDetectorAdapter> trackingDetector;

    cv::Ptr<DetectionBasedTracker> tracker;

    DetectorAgregator(cv::Ptr<CascadeDetectorAdapter> &_mainDetector,
                      cv::Ptr<CascadeDetectorAdapter> &_trackingDetector) :
            mainDetector(_mainDetector),
            trackingDetector(_trackingDetector) {
        CV_Assert(_mainDetector);
        CV_Assert(_trackingDetector);

        DetectionBasedTracker::Parameters DetectorParams;
        tracker = makePtr<DetectionBasedTracker>(mainDetector, trackingDetector, DetectorParams);
    }
};

JNIEXPORT jlong JNICALL
Java_com_jiangdg_opencv4android_natives_DetectionBasedTracker_nativeCreateObject
        (JNIEnv *jenv, jclass, jstring jFileName, jint faceSize) {
    LOGD("Java_org_opencv_samples_facedetect_DetectionBasedTracker_nativeCreateObject enter");
    const char *jnamestr = jenv->GetStringUTFChars(jFileName, NULL);
    string stdFileName(jnamestr);
    jlong result = 0;

    LOGD("Java_org_opencv_samples_facedetect_DetectionBasedTracker_nativeCreateObject");

    try {
        cv::Ptr<CascadeDetectorAdapter> mainDetector = makePtr<CascadeDetectorAdapter>(
                makePtr<CascadeClassifier>(stdFileName));
        cv::Ptr<CascadeDetectorAdapter> trackingDetector = makePtr<CascadeDetectorAdapter>(
                makePtr<CascadeClassifier>(stdFileName));
        result = (jlong) new DetectorAgregator(mainDetector, trackingDetector);
        if (faceSize > 0) {
            mainDetector->setMinObjectSize(Size(faceSize, faceSize));
            //trackingDetector->setMinObjectSize(Size(faceSize, faceSize));
        }
    }
    catch (cv::Exception &e) {
        LOGD("nativeCreateObject caught cv::Exception: %s", e.what());
        jclass je = jenv->FindClass("org/opencv/core/CvException");
        if (!je)
            je = jenv->FindClass("java/lang/Exception");
        jenv->ThrowNew(je, e.what());
    }
    catch (...) {
        LOGD("nativeCreateObject caught unknown exception");
        jclass je = jenv->FindClass("java/lang/Exception");
        jenv->ThrowNew(je,
                       "Unknown exception in JNI code of DetectionBasedTracker.nativeCreateObject()");
        return 0;
    }

    LOGD("Java_org_opencv_samples_facedetect_DetectionBasedTracker_nativeCreateObject exit");
    return result;
}

JNIEXPORT void JNICALL
Java_com_jiangdg_opencv4android_natives_DetectionBasedTracker_nativeDestroyObject
        (JNIEnv *jenv, jclass, jlong thiz) {
    LOGD("Java_org_opencv_samples_facedetect_DetectionBasedTracker_nativeDestroyObject");

    try {
        if (thiz != 0) {
            ((DetectorAgregator *) thiz)->tracker->stop();
            delete (DetectorAgregator *) thiz;
        }
    }
    catch (cv::Exception &e) {
        LOGD("nativeestroyObject caught cv::Exception: %s", e.what());
        jclass je = jenv->FindClass("org/opencv/core/CvException");
        if (!je)
            je = jenv->FindClass("java/lang/Exception");
        jenv->ThrowNew(je, e.what());
    }
    catch (...) {
        LOGD("nativeDestroyObject caught unknown exception");
        jclass je = jenv->FindClass("java/lang/Exception");
        jenv->ThrowNew(je,
                       "Unknown exception in JNI code of DetectionBasedTracker.nativeDestroyObject()");
    }
    LOGD("Java_org_opencv_samples_facedetect_DetectionBasedTracker_nativeDestroyObject exit");
}

JNIEXPORT void JNICALL Java_com_jiangdg_opencv4android_natives_DetectionBasedTracker_nativeStart
        (JNIEnv *jenv, jclass, jlong thiz) {
    LOGD("Java_org_opencv_samples_facedetect_DetectionBasedTracker_nativeStart");

    try {
        ((DetectorAgregator *) thiz)->tracker->run();
    }
    catch (cv::Exception &e) {
        LOGD("nativeStart caught cv::Exception: %s", e.what());
        jclass je = jenv->FindClass("org/opencv/core/CvException");
        if (!je)
            je = jenv->FindClass("java/lang/Exception");
        jenv->ThrowNew(je, e.what());
    }
    catch (...) {
        LOGD("nativeStart caught unknown exception");
        jclass je = jenv->FindClass("java/lang/Exception");
        jenv->ThrowNew(je, "Unknown exception in JNI code of DetectionBasedTracker.nativeStart()");
    }
    LOGD("Java_org_opencv_samples_facedetect_DetectionBasedTracker_nativeStart exit");
}

JNIEXPORT void JNICALL Java_com_jiangdg_opencv4android_natives_DetectionBasedTracker_nativeStop
        (JNIEnv *jenv, jclass, jlong thiz) {
    LOGD("Java_org_opencv_samples_facedetect_DetectionBasedTracker_nativeStop");

    try {
        ((DetectorAgregator *) thiz)->tracker->stop();
    }
    catch (cv::Exception &e) {
        LOGD("nativeStop caught cv::Exception: %s", e.what());
        jclass je = jenv->FindClass("org/opencv/core/CvException");
        if (!je)
            je = jenv->FindClass("java/lang/Exception");
        jenv->ThrowNew(je, e.what());
    }
    catch (...) {
        LOGD("nativeStop caught unknown exception");
        jclass je = jenv->FindClass("java/lang/Exception");
        jenv->ThrowNew(je, "Unknown exception in JNI code of DetectionBasedTracker.nativeStop()");
    }
    LOGD("Java_org_opencv_samples_facedetect_DetectionBasedTracker_nativeStop exit");
}

JNIEXPORT void JNICALL
Java_com_jiangdg_opencv4android_natives_DetectionBasedTracker_nativeSetFaceSize
        (JNIEnv *jenv, jclass, jlong thiz, jint faceSize) {
    LOGD("Java_org_opencv_samples_facedetect_DetectionBasedTracker_nativeSetFaceSize -- BEGIN");

    try {
        if (faceSize > 0) {
            ((DetectorAgregator *) thiz)->mainDetector->setMinObjectSize(Size(faceSize, faceSize));
            //((DetectorAgregator*)thiz)->trackingDetector->setMinObjectSize(Size(faceSize, faceSize));
        }
    }
    catch (cv::Exception &e) {
        LOGD("nativeStop caught cv::Exception: %s", e.what());
        jclass je = jenv->FindClass("org/opencv/core/CvException");
        if (!je)
            je = jenv->FindClass("java/lang/Exception");
        jenv->ThrowNew(je, e.what());
    }
    catch (...) {
        LOGD("nativeSetFaceSize caught unknown exception");
        jclass je = jenv->FindClass("java/lang/Exception");
        jenv->ThrowNew(je,
                       "Unknown exception in JNI code of DetectionBasedTracker.nativeSetFaceSize()");
    }
    LOGD("Java_org_opencv_samples_facedetect_DetectionBasedTracker_nativeSetFaceSize -- END");
}


JNIEXPORT void JNICALL Java_com_jiangdg_opencv4android_natives_DetectionBasedTracker_nativeDetect
        (JNIEnv *jenv, jclass, jlong thiz, jlong imageGray, jlong faces) {
    LOGD("Java_org_opencv_samples_facedetect_DetectionBasedTracker_nativeDetect");

    try {
        vector<Rect> RectFaces;
        ((DetectorAgregator *) thiz)->tracker->process(*((Mat *) imageGray));
        ((DetectorAgregator *) thiz)->tracker->getObjects(RectFaces);
        *((Mat *) faces) = Mat(RectFaces, true);
    }
    catch (cv::Exception &e) {
        LOGD("nativeCreateObject caught cv::Exception: %s", e.what());
        jclass je = jenv->FindClass("org/opencv/core/CvException");
        if (!je)
            je = jenv->FindClass("java/lang/Exception");
        jenv->ThrowNew(je, e.what());
    }
    catch (...) {
        LOGD("nativeDetect caught unknown exception");
        jclass je = jenv->FindClass("java/lang/Exception");
        jenv->ThrowNew(je, "Unknown exception in JNI code DetectionBasedTracker.nativeDetect()");
    }
    LOGD("Java_org_opencv_samples_facedetect_DetectionBasedTracker_nativeDetect END");
}

JNIEXPORT void JNICALL
Java_com_jiangdg_opencv4android_natives_DetectionBasedTracker_saveFisherFaceRecognizer
        (JNIEnv *env, jclass) {
    //读取你的CSV文件路径.
    string fn_csv = "/mnt/sdcard/at.txt";

    // 读取数据. 如果文件不合法就会出错
    // 输入的文件名已经有了.
    try {
        images.clear();
        labels.clear();
        read_csv(fn_csv, images, labels, ';');
    }
    catch (Exception &e) {
        LOGE("Error opening file %s . Reason: %s", fn_csv.c_str(), e.msg.c_str());
        return;
    }
    // 如果没有读取到足够图片，也退出.
    if (images.size() <= 1) {
        LOGE("-->>This demo needs at least 2 images to work. Please add more images to your data set!");
        return;
    }
    LOGE("-->>%zu", images.size());

    // 下面的几行代码仅仅是从你的数据集中移除最后一张图片
    //[gm:自然这里需要根据自己的需要修改，他这里简化了很多问题]
    Mat testSample = images[images.size() - 1];
    int testLabel = labels[labels.size() - 1];
//    images.pop_back();
//    labels.pop_back();

    //保存训练数据
    Ptr<FisherFaceRecognizer> model = FisherFaceRecognizer::create();
    LOGE("-->>fisher训练开始");
    model->train(images, labels);
    LOGE("-->>fisher训练结束");
    LOGE("-->>保存fisher训练集开始");
    model->save("/mnt/sdcard/MyFaceFisherModel.xml");
    LOGE("-->>保存fisher训练集结束");
    globalFisherModel = model;

//    int predictedLabel = model->predict(testSample);
//    LOGE("-->>Predicted name = %d / Actual name = %d.", predictedLabel, testLabel);
    return;
}

JNIEXPORT void JNICALL
Java_com_jiangdg_opencv4android_natives_DetectionBasedTracker_loadFisherFaceRecognizer
        (JNIEnv *env, jclass) {
    //获取训练数据
    fstream _file;
    _file.open("/mnt/sdcard/MyFaceFisherModel.xml", ios::in);
    if (!_file) {
        LOGE("-->>没有fisher训练数据");
    } else {
        LOGE("-->>读取fisher训练数据开始");
        globalFisherModel = Algorithm::load<FisherFaceRecognizer>(
                "/mnt/sdcard/MyFaceFisherModel.xml");
        LOGE("-->>读取fisher训练数据结束");
    }
}

JNIEXPORT jstring JNICALL
Java_com_jiangdg_opencv4android_natives_DetectionBasedTracker_fisherFaceRecognizer
        (JNIEnv *env, jclass, jlong inputImage) {

    imwrite("/mnt/sdcard/fisher.jpg", *((Mat *) inputImage));

    // 下面对测试图像进行预测，predictedLabel是预测标签结果
    if (globalFisherModel == NULL) {
        return env->NewStringUTF("");
    }
    int predictedLabel;
    try {
        //predict中参数mat需为gray格式
        predictedLabel = globalFisherModel->predict(*((Mat *) inputImage));
    } catch (Exception &e) {
        LOGE("-->>fisher识别失败");
        predictedLabel = -1;
    }
    string result_message = format("Predicted name = %d", predictedLabel);
    return env->NewStringUTF(result_message.c_str());
}

JNIEXPORT void JNICALL
Java_com_jiangdg_opencv4android_natives_DetectionBasedTracker_saveEigenFaceRecognizer
        (JNIEnv *env, jclass) {
    //读取你的CSV文件路径.
    string fn_csv = "/mnt/sdcard/at.txt";

    // 读取数据. 如果文件不合法就会出错
    // 输入的文件名已经有了.
    try {
        images.clear();
        labels.clear();
        read_csv(fn_csv, images, labels, ';');
    }
    catch (Exception &e) {
        LOGE("Error opening file %s . Reason: %s", fn_csv.c_str(), e.msg.c_str());
        return;
    }
    // 如果没有读取到足够图片，也退出.
    if (images.size() <= 1) {
        LOGE("-->>This demo needs at least 2 images to work. Please add more images to your data set!");
        return;
    }
    LOGE("-->>%zu", images.size());

//    Mat testSample = images[images.size() - 1];
//    int testLabel = labels[labels.size() - 1];
//    images.pop_back();
//    labels.pop_back();

    //保存训练数据
    Ptr<EigenFaceRecognizer> model = EigenFaceRecognizer::create();
    LOGE("-->>eigen训练开始");
    model->train(images, labels);
    LOGE("-->>eigen训练结束");
    LOGE("-->>保存eigen训练集开始");
    model->save("/mnt/sdcard/MyFaceEigenModel.xml");
    LOGE("-->>保存eigen训练集结束");
    globalEigenModel = model;

//    int predictedLabel = model->predict(testSample);
//    LOGE("-->>Predicted name = %d / Actual name = %d.", predictedLabel, testLabel);
    return;
}

JNIEXPORT void JNICALL
Java_com_jiangdg_opencv4android_natives_DetectionBasedTracker_loadEigenFaceRecognizer
        (JNIEnv *env, jclass) {
    //获取训练数据
    fstream _file;
    _file.open("/mnt/sdcard/MyFaceEigenModel.xml", ios::in);
    if (!_file) {
        LOGE("-->>没有eigen训练数据");
    } else {
        LOGE("-->>读取eigen训练数据开始");
        globalEigenModel = Algorithm::load<EigenFaceRecognizer>(
                "/mnt/sdcard/MyFaceEigenModel.xml");
        LOGE("-->>读取eigen训练数据结束");
    }
}

JNIEXPORT jstring JNICALL
Java_com_jiangdg_opencv4android_natives_DetectionBasedTracker_eigenFaceRecognizer
        (JNIEnv *env, jclass, jlong inputImage) {

    imwrite("/mnt/sdcard/eigen.jpg", *((Mat *) inputImage));

    // 下面对测试图像进行预测，predictedLabel是预测标签结果
    if (globalEigenModel == NULL) {
        return env->NewStringUTF("");
    }
    int predictedLabel;
    try {
        //predict中参数mat需为gray格式
        predictedLabel = globalEigenModel->predict(*((Mat *) inputImage));
    } catch (Exception &e) {
        LOGE("-->>eigen识别失败");
        predictedLabel = -1;
    }
    string result_message = format("Predicted name = %d", predictedLabel);
    return env->NewStringUTF(result_message.c_str());
}

JNIEXPORT void JNICALL
Java_com_jiangdg_opencv4android_natives_DetectionBasedTracker_saveLBPHFaceRecognizer
        (JNIEnv *env, jclass) {
    //读取你的CSV文件路径.
    string fn_csv = "/mnt/sdcard/at.txt";

    // 读取数据. 如果文件不合法就会出错
    // 输入的文件名已经有了.
    try {
        images.clear();
        labels.clear();
        read_csv(fn_csv, images, labels, ';');
    }
    catch (Exception &e) {
        LOGE("Error opening file %s . Reason: %s", fn_csv.c_str(), e.msg.c_str());
        return;
    }
    // 如果没有读取到足够图片，也退出.
    if (images.size() <= 1) {
        LOGE("-->>This demo needs at least 2 images to work. Please add more images to your data set!");
        return;
    }
    LOGE("-->>%zu", images.size());

    // 下面的几行代码仅仅是从你的数据集中移除最后一张图片
    //[gm:自然这里需要根据自己的需要修改，他这里简化了很多问题]
    Mat testSample = images[images.size() - 1];
    int testLabel = labels[labels.size() - 1];
//    images.pop_back();
//    labels.pop_back();

    //保存训练数据
    Ptr<LBPHFaceRecognizer> model = LBPHFaceRecognizer::create();
    globalLBPHModel = model;
    LOGE("-->>LBPH训练开始");
    model->train(images, labels);
    LOGE("-->>LBPH训练结束");
    LOGE("-->>保存LBPH训练集开始");
    model->save("/mnt/sdcard/MyFaceLBPHModel.xml");
    LOGE("-->>保存LBPH训练集结束");
//    model->setThreshold(0.0);
//    int predictedLabel = model->predict(images[images.size() - 2]);
//    LOGE("-->>Predicted name = %d / Actual name = %d.", predictedLabel, testLabel);
    return;
}


JNIEXPORT void JNICALL
Java_com_jiangdg_opencv4android_natives_DetectionBasedTracker_loadLBPHFaceRecognizer
        (JNIEnv *env, jclass) {
    //获取训练数据
    fstream _file;
    _file.open("/mnt/sdcard/MyFaceLBPHModel.xml", ios::in);
    if (!_file) {
        LOGE("-->>没有LBPH训练数据");
    } else {
        LOGE("-->>读取LBPH训练数据开始");
        globalLBPHModel = Algorithm::load<LBPHFaceRecognizer>(
                "/mnt/sdcard/MyFaceLBPHModel.xml");
        LOGE("-->>读取LBPH训练数据结束");
    }
}

JNIEXPORT jstring JNICALL
Java_com_jiangdg_opencv4android_natives_DetectionBasedTracker_lBPHFaceRecognizer
        (JNIEnv *env, jclass, jlong inputImage) {

    //设置之后返回为-1
//    globalMode->setThreshold(10.0);
    imwrite("/mnt/sdcard/lbph.jpg", *((Mat *) inputImage));

    // 下面对测试图像进行预测，predictedLabel是预测标签结果
    if (globalLBPHModel == NULL) {
        return env->NewStringUTF("");
    }
    int predictedLabel;
    try {
        //predict中参数mat需为gray格式
        predictedLabel = globalLBPHModel->predict(*((Mat *) inputImage));
    } catch (Exception &e) {
        LOGE("-->>LBPH识别失败");
        predictedLabel = -1;
    }
//    LOGE("-->>\tLBPH(radius=%i, neighbors=%i, grid_x=%i, grid_y=%i, threshold=%.2f)",
//         globalLBPHModel->getRadius(),
//         globalLBPHModel->getNeighbors(),
//         globalLBPHModel->getGridX(),
//         globalLBPHModel->getGridY(),
//         globalLBPHModel->getThreshold());
    // We could get the histograms for example:
    vector<Mat> histograms = globalLBPHModel->getHistograms();
    // But should I really visualize it? Probably the length is interesting:
//    LOGE("-->>%zu", histograms[0].total());
    string result_message = format("Predicted name = %d", predictedLabel);
    return env->NewStringUTF(result_message.c_str());
}

void read_csv(const string &filename, vector<Mat> &images, vector<int> &labels,
              char separator = ';') {
    ifstream file(filename.c_str(), ifstream::in);
    if (!file) {
        LOGE("No valid input file was given, please check the given filename.");
        return;
    }
    string line, path, classlabel;
    while (getline(file, line)) {
        stringstream liness(line);
        getline(liness, path, separator);
        getline(liness, classlabel);
        if (!path.empty() && !classlabel.empty()) {
            images.push_back(imread(path, 0));
            labels.push_back(atoi(classlabel.c_str()));
        }
    }
}