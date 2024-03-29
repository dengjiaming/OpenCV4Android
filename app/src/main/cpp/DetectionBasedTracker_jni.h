/* DO NOT EDIT THIS FILE - it is machine generated */
#include <jni.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string.h>
#include <opencv2/face.hpp>
/* Header for class org_opencv_samples_fd_DetectionBasedTracker */

#ifndef _Included_org_opencv_samples_fd_DetectionBasedTracker
#define _Included_org_opencv_samples_fd_DetectionBasedTracker
#ifdef __cplusplus
extern "C" {
#endif
/*
 * Class:     org_opencv_samples_fd_DetectionBasedTracker
 * Method:    nativeCreateObject
 * Signature: (Ljava/lang/String;F)J
 */
JNIEXPORT jlong JNICALL Java_com_jiangdg_opencv4android_natives_DetectionBasedTracker_nativeCreateObject
  (JNIEnv *, jclass, jstring, jint);

/*
 * Class:     org_opencv_samples_fd_DetectionBasedTracker
 * Method:    nativeDestroyObject
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_com_jiangdg_opencv4android_natives_DetectionBasedTracker_nativeDestroyObject
  (JNIEnv *, jclass, jlong);

/*
 * Class:     org_opencv_samples_fd_DetectionBasedTracker
 * Method:    nativeStart
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_com_jiangdg_opencv4android_natives_DetectionBasedTracker_nativeStart
  (JNIEnv *, jclass, jlong);

/*
 * Class:     org_opencv_samples_fd_DetectionBasedTracker
 * Method:    nativeStop
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_com_jiangdg_opencv4android_natives_DetectionBasedTracker_nativeStop
  (JNIEnv *, jclass, jlong);

  /*
   * Class:     org_opencv_samples_fd_DetectionBasedTracker
   * Method:    nativeSetFaceSize
   * Signature: (JI)V
   */
  JNIEXPORT void JNICALL Java_com_jiangdg_opencv4android_natives_DetectionBasedTracker_nativeSetFaceSize
  (JNIEnv *, jclass, jlong, jint);

/*
 * Class:     org_opencv_samples_fd_DetectionBasedTracker
 * Method:    nativeDetect
 * Signature: (JJJ)V
 */
JNIEXPORT void JNICALL Java_com_jiangdg_opencv4android_natives_DetectionBasedTracker_nativeDetect
  (JNIEnv *, jclass, jlong, jlong, jlong);

/*
 * Class:     com_jiangdg_opencv4android_natives_DetectionBasedTracker
 * Method:    SaveFaceRecognizer
 * Signature: ()Ljava/lang/String;
 */
JNIEXPORT void JNICALL
Java_com_jiangdg_opencv4android_natives_DetectionBasedTracker_saveLBPHFaceRecognizer
        (JNIEnv *env, jclass);

/*
 * Class:     com_jiangdg_opencv4android_natives_DetectionBasedTracker
 * Method:    loadLBPHFaceRecognizer
 * Signature: ()V
 */
JNIEXPORT void JNICALL
Java_com_jiangdg_opencv4android_natives_DetectionBasedTracker_loadLBPHFaceRecognizer
        (JNIEnv *, jclass);

/*
 * Class:     com_jiangdg_opencv4android_natives_DetectionBasedTracker
 * Method:    lBPHFaceRecognizer
 * Signature: (J)Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL
Java_com_jiangdg_opencv4android_natives_DetectionBasedTracker_lBPHFaceRecognizer
        (JNIEnv *, jclass, jlong);

/*
 * Class:     com_jiangdg_opencv4android_natives_DetectionBasedTracker
 * Method:    saveEigenFaceRecognizer
 * Signature: ()V
 */
JNIEXPORT void JNICALL
Java_com_jiangdg_opencv4android_natives_DetectionBasedTracker_saveEigenFaceRecognizer
        (JNIEnv *, jclass);

/*
 * Class:     com_jiangdg_opencv4android_natives_DetectionBasedTracker
 * Method:    loadEigenFaceRecognizer
 * Signature: ()V
 */
JNIEXPORT void JNICALL
Java_com_jiangdg_opencv4android_natives_DetectionBasedTracker_loadEigenFaceRecognizer
        (JNIEnv *, jclass);

/*
 * Class:     com_jiangdg_opencv4android_natives_DetectionBasedTracker
 * Method:    EigenFaceRecognizer
 * Signature: (J)Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL
Java_com_jiangdg_opencv4android_natives_DetectionBasedTracker_eigenFaceRecognizer
        (JNIEnv *, jclass, jlong);

/*
 * Class:     com_jiangdg_opencv4android_natives_DetectionBasedTracker
 * Method:    saveFisherFaceRecognizer
 * Signature: ()V
 */
JNIEXPORT void JNICALL
Java_com_jiangdg_opencv4android_natives_DetectionBasedTracker_saveFisherFaceRecognizer
        (JNIEnv *, jclass);

/*
 * Class:     com_jiangdg_opencv4android_natives_DetectionBasedTracker
 * Method:    loadFisherFaceRecognizer
 * Signature: ()V
 */
JNIEXPORT void JNICALL
Java_com_jiangdg_opencv4android_natives_DetectionBasedTracker_loadFisherFaceRecognizer
        (JNIEnv *, jclass);

/*
 * Class:     com_jiangdg_opencv4android_natives_DetectionBasedTracker
 * Method:    FisherFaceRecognizer
 * Signature: (J)Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL
Java_com_jiangdg_opencv4android_natives_DetectionBasedTracker_fisherFaceRecognizer
        (JNIEnv *, jclass, jlong);
#ifdef __cplusplus
}

// 2个容器来存放图像数据和对应的标签
using namespace cv;
using namespace cv::face;
using namespace std;
extern vector<Mat> images;
extern vector<int> labels;
extern Ptr<EigenFaceRecognizer> globalEigenModel;
extern Ptr<FisherFaceRecognizer> globalFisherModel;
extern Ptr<LBPHFaceRecognizer> globalLBPHModel;

void read_csv(const string &filename, vector<Mat> &images, vector<int> &labels,
              char separator);

#endif
#endif
