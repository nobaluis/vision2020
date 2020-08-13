# cmake for OpenCV Cookbook 3rd edition chapter 04
# your opencv/build directory should be in your system PATH

# set minimum required version for cmake
cmake_minimum_required(VERSION 2.8)
project(Tarea02)

# add executable
add_executable(svm_hog svm_hog.cpp)
add_executable(svm_lbp svm_lbp.cpp)
add_executable(svm_lbp_hog svm_lbp_hog.cpp)
add_executable(gen_pos generate_pos.cpp)
add_executable(gen_neg generate_neg.cpp)

# OpenCV4 lib
SET("OpenCV_DIR" "/usr/local/include/opencv4")
find_package(OpenCV REQUIRED)

# FFW3 lib
find_path(FFTW_INCLUDE_DIR fftw3.h  ${FFTW_INCLUDE_DIRS})
find_library(FFTW_LIBRARY fftw3 ${FFTW_LIBRARY_DIRS})

# LBP lib
add_library(LBP STATIC LBP.cpp)
target_link_libraries(LBP ${OpenCV_LIBS} ${FFTW_LIBRARY})

# link libraries
target_link_libraries(svm_hog ${OpenCV_LIBS})
target_link_libraries(svm_lbp ${OpenCV_LIBS} LBP)
target_link_libraries(svm_lbp_hog ${OpenCV_LIBS} LBP)
target_link_libraries(gen_pos ${OpenCV_LIBS})
target_link_libraries(gen_neg ${OpenCV_LIBS})
