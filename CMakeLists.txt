cmake_minimum_required(VERSION 3.16)
project(gauss_cpu LANGUAGES CXX CUDA)     # <-- add CUDA

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)               # <-- optional, good on 3.16
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Tell CMake to use the new behavior for CUDA arch detection
cmake_policy(SET CMP0104 NEW)

# Pick your GPU's SM architecture(s)
# Examples:
# 61 = Pascal (GTX 10xx), 70 = Volta, 75 = Turing (RTX 20xx),
# 86 = Ampere (RTX 30xx), 89 = Ada (RTX 40xx), 90 = Hopper
set(CMAKE_CUDA_ARCHITECTURES 61)   # <-- change to match your GPU

# Default build type = Release
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Find OpenCV (make sure it’s installed)
find_package(OpenCV REQUIRED)

# Add include/ directory to search path
include_directories(${CMAKE_SOURCE_DIR}/include)

# Define the executable and its source files
add_executable(gauss_cpu
  src/main.cpp
  src/blur.cpp
  src/utils.cpp
  src/gauss_cuda.cu           # <-- add your CUDA file
)

# (Optional) pick your GPU arch; comment out if unsure
# set(CMAKE_CUDA_ARCHITECTURES 75)

# (Optional) enable separable compilation
# set_target_properties(gauss_cpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Link OpenCV libraries
target_link_libraries(gauss_cpu PRIVATE ${OpenCV_LIBS})
