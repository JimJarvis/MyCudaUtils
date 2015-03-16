cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
set(CMAKE_LEGACY_CYGWIN_WIN32 0)
project(MyCudaUtils CXX)

find_package(CUDA REQUIRED)

# include_directories( )

# Ugly workaround for CMake-NVCC bug
if (APPLE)
    set(CUDA_HOST_COMPILER /usr/bin/clang++)
elseif(UNIX)
    set(CUDA_HOST_COMPILER /usr/bin/g++)
endif()

set(SOURCES 
    test.cu)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-std=c++11 -O3 -arch=sm_30)

cuda_add_executable(test ${SOURCES})
