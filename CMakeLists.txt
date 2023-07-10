cmake_minimum_required(VERSION 3.10)

project(proceedrizer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(OpenMP)

if(OPENMP_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gl.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Matrix.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Model.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/tgaimage.h)
set(SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gl.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/tgaimage.cpp)

add_executable(proceedrizer ${SOURCES})
