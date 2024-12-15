cmake_minimum_required(VERSION 3.8)
project(cuda_project CUDA CXX)

# Set CUDA architectures based on your GPU
set(CMAKE_CUDA_ARCHITECTURES 86)  # Adjust this for your GPU architecture

# Find CUDA package
find_package(CUDA REQUIRED)

# Add executable
add_executable(cuda_app src/main.cu)
set_target_properties(cuda_app PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Enable debug symbols for development
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G")