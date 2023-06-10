cmake_minimum_required(VERSION 3.25)
project(cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
add_compile_options("-DNOSUBMIT")


add_executable(cpp main.cpp)
