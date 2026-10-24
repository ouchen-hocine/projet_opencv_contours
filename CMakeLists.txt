cmake_minimum_required(VERSION 3.10)
project(homework3)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(main_prog main.cpp)
target_link_libraries(main_prog ${OpenCV_LIBS})
