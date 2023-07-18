cmake_minimum_required(VERSION 3.10)
# Every project needs a name.  We call this the "examples" project.
project(simple-glasses-detector)

# include dlib
add_subdirectory(../dlib dlib_build)

# main component, src files
add_library(eyeglasses-detector STATIC svm_model.hpp eyeglasses_detector.cpp)

# link libs and include dirs
target_link_libraries(eyeglasses-detector PUBLIC dlib::dlib)
target_include_directories(eyeglasses-detector PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Properties
set_target_properties(eyeglasses-detector PROPERTIES PUBLIC_HEADER eyeglasses_detector.h)
set_property(TARGET eyeglasses-detector PROPERTY CXX_STANDARD 20)
set_property(TARGET eyeglasses-detector PROPERTY CXX_STANDARD_REQUIRED ON)

# Installation targets
install(TARGETS eyeglasses-detector 
	CONFIGURATIONS Debug 
	DESTINATION ${CMAKE_SOURCE_DIR}/out/Debug
	PUBLIC_HEADER DESTINATION ${CMAKE_SOURCE_DIR}/out/include
)

install(TARGETS eyeglasses-detector 
	CONFIGURATIONS Release 
	DESTINATION ${CMAKE_SOURCE_DIR}/out/Release
	PUBLIC_HEADER DESTINATION ${CMAKE_SOURCE_DIR}/out/include
)



