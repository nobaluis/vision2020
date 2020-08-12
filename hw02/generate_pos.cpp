/*
 * Computer Vision 3D (2020) - Assignment 02
 * @file generate-pos.cpp
 * @author Luis Castillo <luis.castillo@cinvestav.mx>
 * @version 1.0
*/

#include <iostream>
#include <fstream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>


const std::string base_path("/home/luisc/Git/vision2020/hw02/db");

void savePedestrians(const std::string& city, int img_no, const std::string& target_dir){
    // File name
    char number[5];
    std::sprintf(number, "%05d", img_no);
    std::string src_path = base_path + "/src/pedestrians/" + city + "Ped" + number;
    //std::string img_path = src_path + ".png";
    // Read the image
    cv::Mat img = cv::imread(src_path+".png");
    // Read the bounding box (bb) pedestrian coordinates
    std::string line;
    std::ifstream textfile(src_path+".txt");
    if(textfile.is_open()){
        int i = 0;
        while( getline(textfile, line)){
            // bb coordinates
            int x1, y1, x2, y2;
            std::sscanf(line.c_str(), "(%d,%d) - (%d,%d)",&x1,&y1,&x2,&y2);
            // build the bb
            cv::Rect bb(x1,y1,x2-x1,y2-y1);
            // read the pedestrian
            cv::Mat ped = img(bb);
            // resize image
            cv::Mat ped_norm;
            cv::resize(ped, ped_norm, cv::Size(64,128));
            // save resize pedstrian
            std::stringstream ped_path;
            ped_path << base_path << "/pos/" << target_dir << '/' << city << '_' << img_no << '_' << i;
            cv::imwrite(ped_path.str()+".png", ped_norm);
            ++i;
        }
        textfile.close();
    }
}


int main(){
    // Generate Fudan dataset
    // FudanPed000xx, training[1, 60], testing[61, 74]
    for(int i = 1; i < 75; ++i){
        if(i < 61){
            savePedestrians("Fudan", i, "training");
        } else{
            savePedestrians("Fudan", i, "testing");
        }
    }
    // Generate Penn dataset
    // PennPed000xx, training[1, 70], testing[71, 96]
    for(int i = 1; i < 97; ++i){
        if(i < 71){
            savePedestrians("Penn", i, "training");
        } else{
            savePedestrians("Penn", i, "testing");
        }
    }
    return 0;
}

