/*
 * Computer Vision 3D (2020) - Assignment 02
 * @file generate-neg.cpp
 * @author Luis Castillo <luis.castillo@cinvestav.mx>
 * @version 1.0
*/

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>


const std::string base_path("/home/luisc/Git/vision2020/hw02/db");


void saveRandomBB(const std::string& file_name, int samples, const std::string& target_dir){
    //read image
    cv::Mat img = cv::imread(base_path+"/src/non_pedestrains/"+file_name+".png");
    printf("original size = {w: %d, h:%d}\n",img.cols,img.rows);

    // generate random bb's
    for(int i=0; i < samples; ++i){
        //random bb
        int width = (std::rand() % (img.cols / 2)) + 50;
        int height = (std::rand() % (img.rows / 2)) + 50;
        int x = (std::rand() % (img.cols - width)) + 1;
        int y = (std::rand() % (img.rows - height)) + 1;
        printf("BB = { x: %d, y: %d, w: %d, h: %d}\n",x,y,width,height);

        // build bb¸
        cv::Rect bb(x, y, width, height);

        // extract ROI
        cv::Mat roi = img(bb);

        // resizezSe
        cv::Mat roi_norm;
        cv::resize(roi, roi_norm, cv::Size(64, 128));

        // save
        std::stringstream file_path;
        file_path << base_path << "/neg/" << target_dir << '/' << file_name << '_' << i << ".png";
        cv::imwrite(file_path.str(), roi_norm);
    }
}

int main(){
    //Generate training data [1-4]
    for(int i = 1; i < 5; ++i){
        saveRandomBB(std::to_string(i), 150, "training");
    }
    //Generate testing data [5-6]
    for(int i = 5; i < 7; ++i){
        saveRandomBB(std::to_string(i), 80, "testing");
    }
	return 0;
}

