/*
 * Computer Vision 3D (2020) - Assignment 02
 * @file main.cpp
 * @author Luis Castillo <luis.castillo@cinvestav.mx>
 * @version 1.0
*/

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/ml.hpp>


const std::string base_path = "/home/luisc/Git/vision2020/hw02/db";

/**
 * @brief pushHog Compute de hog descriptor for given Mat image and store in a Mat
 * @param hogDesc The HOGDescriptor object
 * @param desc Vector to store the HOG descriptor
 * @param srcImg The image in a Mat object
 * @param dest The Mat where the descriptor will be stored
 */
void pushHog(cv::HOGDescriptor& hogDesc, std::vector<float> desc, cv::Mat& srcImg, cv::Mat& dest){
    hogDesc.compute(srcImg, desc);
    dest.push_back(cv::Mat(desc, true).reshape(1,1));
}

/**
 * @brief computeHog Compute HOG descriptor for given filename and store in a Mat
 * @param hogDesc The HOGDescriptor object
 * @param desc Vector to store the HOG descriptor
 * @param srcFile The filename path where is the image
 * @param srcImg The Mat object for store the image
 * @param dest The Mat where the descriptors will be stored
 * @param augmented If this is true, the image is fliped to get X2 descriptors
 */
void computeHog(cv::HOGDescriptor& hogDesc, std::vector<float> desc,
                const cv::String& srcFile, cv::Mat& srcImg, cv::Mat& dest, bool augmented){
    srcImg = cv::imread(srcFile, cv::IMREAD_GRAYSCALE);
    pushHog(hogDesc, desc, srcImg, dest);
    if(augmented){
        cv::flip(srcImg, srcImg, 1);
        pushHog(hogDesc, desc, srcImg, dest);
    }
}


int main(){
    // PRELIMINARIES

    // 1. Declare the variables that will be used in the program
    cv::Mat img, samples, test_samples, predictions;
    std::vector<float> descriptor;
    std::vector<cv::String> positives, negatives, positives_test, negatives_test;

    // 2. Create the HOG descriptor object
    cv::HOGDescriptor hog(cv::Size(64, 128),        // win_size
                              cv::Size(8, 8),       // block_size
                              cv::Size(4, 4),       // block_stride
                              cv::Size(4, 4), 9);   // cell_size, nbins


    // PART I - Process the trainning data

    // 1. Read all files in the training directory
    cv::glob(base_path+"/pos/training/*.png", positives, false);
    cv::glob(base_path+"/neg/training/*.png", negatives, false);

    // 2. Generate the samples with HOG descriptors
    // 2.1. Compute positives descriptors (with data augmentation)
    for(auto file : positives){
        computeHog(hog, descriptor, file, img, samples, true);
    }
    // 2.2. Compute negatives descriptors
    for(auto file : negatives){
        computeHog(hog, descriptor, file, img, samples, false);
    }

    // 3. Create the labels
    cv::Mat labels(samples.rows, 1, CV_32SC1);
    labels.rowRange(0, positives.size()*2) = 1.0;
    labels.rowRange(positives.size()*2, samples.rows) = -1.0;


    // PART II - Train the SVM

    // 1. Prepare the training data
    cv::Ptr<cv::ml::TrainData> traning_data =
            cv::ml::TrainData::create(samples, cv::ml::SampleTypes::ROW_SAMPLE, labels);

    // 2. Create the svm and train with the training data
    cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
    svm->setType(cv::ml::SVM::C_SVC);
    svm->setKernel(cv::ml::SVM::LINEAR);
    svm->train(traning_data);


    // PART III - Test the SVM

    // 1. Read all files in the testing directory
    cv::glob(base_path+"/pos/testing/*.png", positives_test, false);
    cv::glob(base_path+"/neg/testing/*.png", negatives_test, false);

    // 2. Generate the testing samples with HOG descriptors
    // 2.1. Compute HOG descriptors for the positives
    for(auto file: positives_test){
        computeHog(hog, descriptor, file, img, test_samples, false);
    }
    // 2.2. Compute HOG descriptors for the negatives
    for(auto file: negatives_test){
        computeHog(hog, descriptor, file, img, test_samples, false);
    }

    // 3. Do the predictions
    svm->predict(test_samples, predictions);

    // 4. Verify the results
    int pos_fails = 0;
    int neg_fails = 0;
    for(int i = 0; i < predictions.rows; ++i){
        // printf("Prediction at %03d is %.0f\n", i, predictions.at<float>(i));
        if(i < int(positives_test.size())){
            //these elements should be positive
            if(predictions.at<float>(i) < 0.0){
                ++pos_fails;
            }

        } else {
            //these elements should be negative
            if(predictions.at<float>(i) > 0.0){
                ++neg_fails;
            }
        }
    }

    // 5. Compute accuracy
    float pos_acc, neg_acc, acc;
    pos_acc = (positives_test.size() * 2 - pos_fails) / float(positives_test.size() * 2);
    neg_acc = (negatives_test.size() - neg_fails) / float(negatives_test.size());
    acc = (test_samples.rows - pos_fails - neg_fails) / float(test_samples.rows);

    printf("HOG SVM Pedestrians Detector \n");
    printf("Positives accuracy = %.5f\n", pos_acc);
    printf("Negatives accuracy = %.5f\n", neg_acc);
    printf("Total accuracy = %.5f\n", acc);

	return 0;
}
