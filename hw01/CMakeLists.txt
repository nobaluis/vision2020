# cmake for OpenCV Cookbook 3rd edition chapter 04
# your opencv/build directory should be in your system PATH

# set minimum required version for cmake
cmake_minimum_required(VERSION 2.8)
project(Tarea01)

# add executable
add_executable(main main.cpp)

# link libraries
SET("OpenCV_DIR" "/usr/local/include/opencv4")
find_package(OpenCV REQUIRED)
target_link_libraries(main ${OpenCV_LIBS})

# copy required images to every directory with executable
SET (IMAGES ${CMAKE_SOURCE_DIR}/../images/shape_test.jpg)
SET (IMAGES ${CMAKE_SOURCE_DIR}/../images/all_shapes.jpg)
FILE(COPY ${IMAGES} DESTINATION .)
FILE(COPY ${IMAGES} DESTINATION "Debug")
FILE(COPY ${IMAGES} DESTINATION "Release")
