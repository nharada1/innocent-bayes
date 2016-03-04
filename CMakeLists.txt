cmake_minimum_required(VERSION 3.3)
project(innocent_bayes)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES main.cpp NaiveBayes.cpp NaiveBayes.h)
add_executable(innocent_bayes ${SOURCE_FILES})