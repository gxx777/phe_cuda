cmake_minimum_required(VERSION 3.6)

project(gpu_lib)

find_package(CUDA)

set(CMAKE_CXX_STANDARD 14)

set(PYTHON_INCLUDE_DIRS "/root/miniconda3/include/python3.8")
set(PYTHON_LIBRARIES "/root/miniconda3/lib")
set(CGBN_INCLUDES "/root/CGBN/include")
set(PYBIND11_INCLUDE "/root/pybind11/include")


SET(CUDA_NVCC_FLAGS -arch=sm_61)


include_directories(
  ${PYTHON_INCLUDE_DIRS}
  ${PYBIND11_INCLUDE}
  ${CGBN_INCLUDES}
  )

link_directories(
  /usr/local/cuda/lib64
  ${PYTHON_LIBRARIES}
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

file(GLOB CURRENT_SOURCES  powm_mulm.cu)

cuda_add_library(gpu_lib SHARED  ${CURRENT_SOURCES})


target_link_libraries(gpu_lib cudart)

target_link_libraries(gpu_lib gmp gmpxx)

set_target_properties(gpu_lib PROPERTIES PREFIX "")
