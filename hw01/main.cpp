/*
 * Computer Vision 3D (2020) - Assignment 01
 * @file main.cpp
 * @author Luis Castillo <luis.castillo@cinvestav.mx>
 * @version 1.0
 *
 * Assignment description:
 * 1. Load image in gray scale
 * 2. Binarize image nicely  (use possible filters, blur, morpho, etc..)
 * 3. Extract contours
 * 4. Extract centroids
 * 5. Compute BOF (Boundary Object Function)
 * 6. Normalize size and scale of BOF 180 elements distance: [0 to 1.0]
 * 7. Create a DataSet with all shapes /save to file to plot
 * 8. Load a cropped image of a single shape
 * 9. Extract normalized BOF of loaded shape
 * 10. Compare with all in data set, and tell wich one it is
*/

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>


void display_image(cv::Mat &img, const std::string& windowname){
    cv::namedWindow(windowname);
    cv::imshow(windowname, img);
}

void draw_contours(cv::Mat& img, std::vector<std::vector<cv::Point>>& contours){
    cv::drawContours(img,contours,-1,cv::Scalar(0,0,255),3);
}

/**
 * @brief draw_centroids Draw cirlces at centroids positions and optionally a label
 * @param img The image to draw
 * @param centroids Centroids vector
 * @param draw_index If is true a label is drawed
 */
void draw_centroids(cv::Mat& img, std::vector<cv::Point>& centroids, bool draw_index){
    for(uint i=0; i < centroids.size(); ++i){
        cv::circle(img,centroids.at(i),3,cv::Scalar(0,255,0),-1);
        if(draw_index){
            cv::putText(img,std::to_string(i),centroids.at(i),cv::FONT_HERSHEY_DUPLEX,1.0,cv::Scalar(255,0,0));
        }
    }
}

/**
 * @brief binarize Binarize an gray scale iamge with threshold
 * @param input The input image in grayscale
 * @param output The output binary image
 */
void binarize(cv::Mat& input, cv::Mat& output){
    cv::threshold(input, output, 245, 255, cv::THRESH_BINARY_INV);
    // remove little shapes with morph filters
    cv::Mat square9(9,9,CV_8U,cv::Scalar(1));
    cv::Mat square7(7,7,CV_8U,cv::Scalar(1));
    cv::Mat square5(5,5,CV_8U,cv::Scalar(1));
    cv::erode(output,output,square9);
    cv::erode(output,output,square7);
    cv::morphologyEx(output,output,cv::MORPH_OPEN,square5);
}

/**
 * @brief extract_contours Extract countours in image
 * @param input The image to analize
 * @param output The contours list, 2d vector of cv points
 * @param min_size Used to discard contours
 */
void extract_contours(cv::Mat& input, std::vector<std::vector<cv::Point>>& output, const int& min_size){
    // find countours
    cv::findContours(input,output,cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
    // filter by size (if is needed)
    if(min_size > 0){
        for(auto it = output.begin(); it != output.end(); ++it){
            if((int)it->size() < min_size){
                output.erase(it);
            }
        }
    }
}

/**
 * @brief extract_centroids Computes the centroids of contours list
 * @param input The contours list, 2d vector of cv points
 * @param output The centroids list, 1d vector of cv points
 */
void extract_centroids(std::vector<std::vector<cv::Point>>& input, std::vector<cv::Point>& output){
    // compute centroids
    for(auto& contour : input){
        int cx=0, cy=0;
        for(auto& point : contour){
            cx += point.x;
            cy += point.y;
        }
        cx /= contour.size();
        cy /= contour.size();
        output.push_back(cv::Point(cx,cy));
    }
}

/**
 * @brief compute_bofs Computes the BOF (Boundary Object Function)
 * @param contours The list of contours, 2d vector of cv points
 * @param centroids The list of centroids, 1d vector of cv points
 * @param output The BOF of each pair of (contour,centroid), 2d vector of doubles
 */
void compute_bofs(std::vector<std::vector<cv::Point>>& contours, std::vector<cv::Point>& centroids,
                 std::vector<std::vector<double>>& output){
    std::vector<double> distances;
    for(uint i = 0; i < centroids.size(); ++i){
        std::vector<cv::Point> contour = contours.at(i);
        cv::Point centroid = centroids.at(i);
        for(auto& point : contour){
            double dis = cv::norm(centroid - point);
            distances.push_back(dis);
        }
        output.push_back(distances);
        distances.clear();
    }
}

//// Computes the max value in a vector of doubles
double max_value(std::vector<double>& array){
    double max = -1;
    for(auto& val : array){
        if(val > max){
            max = val;
        }
    }
    return max;
}

//// Normalize and resize each BOF in list
void normalize_bofs(std::vector<std::vector<double>>& input){
    double max;
    for(auto& bof: input){
       max = max_value(bof); //find max value
       for(auto& dis : bof){
           dis /= max; // normalize each value
       }
       cv::resize(bof,bof,cv::Size(180,1)); // resize (N,1) -> (180,1)
    }
}

//// Save BOF list in a file
void save_bofs(std::vector<std::vector<double>>& input, const std::string& filename){
    cv::FileStorage file(filename, cv::FileStorage::WRITE);
    file << "BOF_LIST" << input;
}


//// Read BOF list from file
void load_bofs(const std::string& filename, std::vector<std::vector<double>>& output){
    cv::FileStorage file(filename, cv::FileStorage::READ);
    file["BOF_LIST"] >> output;
}

//// Computes the correlation coeffient bweeten two vectors
double correlation(std::vector<double>& x, std::vector<double>& y){
    int n = x.size();
    double x_sum = 0, y_sum = 0, xy_sum = 0;
    double x_ssum = 0, y_ssum = 0;
    for(int i=0; i < n; ++i){
        x_sum += x.at(i);
        y_sum += y.at(i);
        xy_sum += x.at(i) * y.at(i);
        x_ssum += x.at(i) * x.at(i);
        y_ssum += y.at(i) * y.at(i);
    }
    double corr = (double)(n*xy_sum-x_sum*y_sum)/
            sqrt((n*x_ssum-x_sum*x_sum)*(n*y_ssum-y_sum*y_sum));
    return corr;
}

/**
 * @brief find_bof Find the most correlated BOF in a list of BOF's
 * @param bof_list The list of BOF, 2d vector of doubles
 * @param bof The BOF to be found
 * @return The index of most correlated BOF in a bof_list
 */
int find_bof(std::vector<std::vector<double>>& bof_list, std::vector<double>& bof){
    int n = bof_list.size();
    int max_index = 0;
    double max_corr = -1;
    for(int i = 0; i < n; ++i){
        double crrt_corr = correlation(bof_list.at(i), bof);
        if(crrt_corr > max_corr){
            max_corr = crrt_corr;
            max_index = i;
        }
    }
    return max_index;
}


int main(){
    // PART I - Generate BOF database


    //Step 1 - load image in gray scale
    cv::Mat img, img_c1;
    img = cv::imread("all_shapes.jpg", cv::IMREAD_COLOR); // load in color for viz
    cv::cvtColor(img,img_c1,cv::COLOR_BGR2GRAY); // bgr -> grayscale

    //Step 2 - binarize
    binarize(img_c1,img_c1);

    //Step 3 - extract countours
    std::vector<std::vector<cv::Point>> contours;
    extract_contours(img_c1,contours,0);
    draw_contours(img,contours);
    std::cout << "Contours found: " << contours.size() << '\n';

    //Step 4 - find centroids
    std::vector<cv::Point> centroids;
    extract_centroids(contours, centroids);
    draw_centroids(img,centroids,true);

    //Step 5 - compute BOF
    std::vector<std::vector<double>> bof_list;
    compute_bofs(contours, centroids, bof_list);

    //Step 6 - normalize BOF
    normalize_bofs(bof_list);

    //Step 7 - create dataset
    save_bofs(bof_list, "bof_list.xml");


    // PART II - Load single shape image and find BOF class in db


    //Setp 8 - load cropped shape
    cv::Mat shape, shape_c1;
    shape = cv::imread("shape_test.jpg", cv::IMREAD_COLOR); // load in color for viz
    cv::cvtColor(shape,shape_c1,cv::COLOR_BGR2GRAY); // bgr -> grayscale
    binarize(shape_c1,shape_c1); // grayscale -> binary

    //Step 9 - extract and normalize bof
    std::vector<std::vector<cv::Point>> shape_contours;
    std::vector<cv::Point> shape_centroids;
    std::vector<std::vector<double>> shape_bofs;

    extract_contours(shape_c1,shape_contours,0);
    draw_contours(shape, shape_contours);
    extract_centroids(shape_contours, shape_centroids);
    draw_centroids(shape,shape_centroids,false);
    compute_bofs(shape_contours, shape_centroids, shape_bofs);
    normalize_bofs(shape_bofs);

    //Step 10 - compare shape bof to bof's db
    std::vector<std::vector<double>> restore_bofs;
    load_bofs("bof_list.xml", restore_bofs); //restore db from file
    int bof_index = find_bof(restore_bofs, shape_bofs.at(0)); // find best match in db
    cv::putText(shape, std::to_string(bof_index),
                shape_centroids.at(0),
                cv::FONT_HERSHEY_DUPLEX,
                1.0,cv::Scalar(255,0,0)); // draw result
    std::cout << "Match bof class is " << bof_index << '\n';

    // Display images
    display_image(img, "All shapes (original)");
    display_image(img_c1, "All shapes (binary)");
    display_image(shape, "Test shape (original)");
    display_image(shape_c1, "Test shape (binary)");
    cv::waitKey(0); // wait for key to close
    cv::destroyAllWindows(); // destroy all windows
	return 0;
}
