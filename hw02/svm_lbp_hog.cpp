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

#include "LBP.hpp"
#include "LBPGPU.cuh"

const std::string base_path = "/home/luisc/Git/vision2020/hw02/db";



void computeHogDesc(cv::HOGDescriptor& hogObj, std::vector<float>& desc, cv::Mat& srcImg){
    hogObj.compute(srcImg, desc);
}

void computeLbpDesc(lbp::LBP& lbpObj, std::vector<float>& desc, std::vector<double>& hist,
                    cv::Mat& srcImg){
    lbpObj.calcLBP(srcImg, 1, true);  // compute LBP
    hist = lbpObj.calcHist().getHist(true); // compute the histogram
    desc.assign(hist.begin(), hist.end());  // cast to float
}


void computeDesc(cv::HOGDescriptor& hogObj, lbp::LBP& lbpObj,
                 std::vector<float>& hogDesc, std::vector<float>& lbpDesc, std::vector<double>& hist,
                 cv::Mat& srcImg, cv::Mat& srcImgF, cv::Mat& dest){
    // 1. Compute the HOG descriptor
    computeHogDesc(hogObj, hogDesc, srcImg);

    // test - normalize hogDesc
    cv::normalize(hogDesc, hogDesc);


    // 2. Compute the LBP-HF descriptor
    computeLbpDesc(lbpObj, lbpDesc, hist, srcImgF);
    // 3. Concatenate vectors
    std::vector<float> fullDesc;
    std::copy(hogDesc.begin(), hogDesc.end(), back_inserter(fullDesc));
    std::copy(lbpDesc.begin(), lbpDesc.end(), back_inserter(fullDesc));
    // 4. Push to samples data
    dest.push_back(cv::Mat(fullDesc, true).reshape(1,1));
}


void processSample(cv::HOGDescriptor& hogObj, lbp::LBP& lbpObj,
                   std::vector<float>& hogDesc, std::vector<float>& lbpDesc, std::vector<double>& hist,
                   cv::Mat& srcImg, cv::Mat& srcImgF, cv::Mat& dest,
                   const cv::String& srcFile, bool augmented){
    srcImg = cv::imread(srcFile, cv::IMREAD_GRAYSCALE);
    srcImg.convertTo(srcImgF, CV_64F);
    computeDesc(hogObj, lbpObj, hogDesc, lbpDesc, hist, srcImg, srcImgF, dest);
    if(augmented){
        cv::flip(srcImg, srcImg, 1);
        srcImg.convertTo(srcImgF, CV_64F);
        computeDesc(hogObj, lbpObj, hogDesc, lbpDesc, hist, srcImg, srcImgF, dest);
    }
}


int main(){
    // PRELIMINARIES

    // 1. Declare the variables that will be used in the program
    std::vector<double> lbp_hist;
    std::vector<float> lbp_desc, hog_desc;
    cv::Mat img, img_f, samples, test_samples, predictions;
    std::vector<cv::String> positives, negatives, positives_test, negatives_test;

    // 2. Create the LPB obj
    lbp::LBP lbp(16, lbp::LBP_MAPPING_HF);

    // 3. Create the HOGDesc obj
    cv::HOGDescriptor hog(cv::Size(64, 128),        // win_size
                              cv::Size(8, 8),       // block_size
                              cv::Size(4, 4),       // block_stride
                              cv::Size(4, 4), 9);   // cell_size, nbins

    // PART I - Process the trainning data

    // 1. Read all files in the training directory
    cv::glob(base_path+"/pos/training/*.png", positives, false);
    cv::glob(base_path+"/neg/training/*.png", negatives, false);


    /*processSample(hog, lbp, hog_desc,
                  lbp_desc, lbp_hist,
                  img, img_f, test_samples,
                  positives[0], false);*/

    // 2. Generate the samples with HOG descriptors
    // 2.1. Compute positives descriptors (with data augmentation)
    for(auto file : positives){
        processSample(hog, lbp, hog_desc,
                      lbp_desc, lbp_hist,
                      img, img_f, samples,
                      file, true);
    }
    // 2.2. Compute negatives descriptors
    for(auto file : negatives){
        processSample(hog, lbp, hog_desc,
                      lbp_desc, lbp_hist,
                      img, img_f, samples,
                      file, false);
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
        processSample(hog, lbp, hog_desc,
                      lbp_desc, lbp_hist,
                      img, img_f, test_samples,
                      file, false);
    }
    // 2.2. Compute HOG descriptors for the negatives
    for(auto file: negatives_test){
        processSample(hog, lbp, hog_desc,
                      lbp_desc, lbp_hist,
                      img, img_f, test_samples,
                      file, false);
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

    float pos_acc, neg_acc, acc;
    pos_acc = (positives.size() * 2 - pos_fails) / float(positives.size() * 2);
    neg_acc = (negatives.size() - neg_fails) / float(negatives.size());
    acc = (samples.rows - pos_fails - neg_fails) / float(samples.rows);

    printf("LBP+HOG SVM Pedestrians Detector\n");
    printf("Positives accuracy = %.5f\n", pos_acc);
    printf("Negatives accuracy = %.5f\n", neg_acc);
    printf("Total accuracy = %.5f\n", acc);

    return 0;
}
