cmake_minimum_required(VERSION 3.17.0)
project(blurhash VERSION 1.0 LANGUAGES C CXX CUDA)

#==========================================
#   BlurHashCPU
#==========================================

file(GLOB_RECURSE BlurHash_CPU_SRC "blurhash/*.cpp" "blurhash/*.h")

add_executable(blurhashCPU ${BlurHash_CPU_SRC})

set_property(TARGET blurhashCPU PROPERTY CXX_STANDARD 17)

#==========================================
#   BlurHashGPU
#==========================================

file(GLOB_RECURSE BlurHash_GPU_SRC "blurhash/*.cu" "blurhash/*.h")

add_executable(blurhashGPU ${BlurHash_GPU_SRC})

set_property(TARGET blurhashGPU PROPERTY CXX_STANDARD 17)