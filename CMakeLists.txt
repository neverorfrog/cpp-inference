cmake_minimum_required(VERSION 3.19)

# Check for build types
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE
      Release
      CACHE STRING "" FORCE)
endif()
message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")

# Compile options
add_library(cxx_setup INTERFACE)
target_compile_features(cxx_setup INTERFACE cxx_std_17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
if(CMAKE_CXX_COMPILER_LOADED)
  message(STATUS "Path: ${CMAKE_CXX_COMPILER}")
  message(STATUS "Compiler ID: ${CMAKE_CXX_COMPILER_ID}")
  message(STATUS "Compiler Version: ${CMAKE_CXX_COMPILER_VERSION}")
endif()
set(CMAKE_C_COMPILER "/usr/bin/gcc")
set(CMAKE_CXX_COMPILER "/usr/bin/g++")
set(EXECUTABLE_OUTPUT_PATH "${CMAKE_CURRENT_BINARY_DIR}/executables")

# just for printing
string(ASCII 27 Esc)
set(ColourReset "${Esc}[m")
set(Red "${Esc}[31m")
set(Green "${Esc}[32m")
set(Blue "${Esc}[34m")

project(et-demo VERSION 0.1 LANGUAGES CXX)

# -------- Build Executorch -----------------------------------
set(PYTHON_EXECUTABLE python)
option(EXECUTORCH_BUILD_XNNPACK "" ON) # Build with Xnnpack backend
option(EXECUTORCH_BUILD_EXTENSION_DATA_LOADER "" ON)
option(EXECUTORCH_BUILD_EXTENSION_MODULE "" ON)
option(EXECUTORCH_BUILD_OPTIMIZED "" ON)
set(EXECUTORCH_ENABLE_LOGGING 1)

set(ET_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/external/executorch)
set(ET_BUILD_PATH ${CMAKE_BINARY_DIR}/external/executorch)

add_subdirectory(
    ${ET_ROOT}
    ${ET_BUILD_PATH}
)
# --------------------------------------------------------------

# ---- For some reason extension_runner_util must be found -----
link_directories("${ET_ROOT}/cmake-out/extension/runner_util")
find_library(UTIL_LIB extension_runner_util HINTS "${ET_ROOT}/cmake-out/extension/runner_util")
if(NOT UTIL_LIB)
  message(FATAL_ERROR "runner_util library not found")
endif()
# --------------------------------------------------------------

# -------- Find libtorch ---------------------------------------
list(APPEND CMAKE_PREFIX_PATH "external/libtorch")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
if(Torch_FOUND)
  message(STATUS "${Green}LIB_Torch found${ColourReset}")
endif()
# --------------------------------------------------------------

# ----------- Find onnxruntime ---------------------------------
set(ONNX_ROOT "${CMAKE_CURRENT_LIST_DIR}/external/onnxruntime/build/Linux/RelWithDebInfo/libonnxruntime.so")
message(STATUS "onnxruntime_ROOT: ${ONNX_ROOT}")
add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES IMPORTED_LOCATION "${ONNX_ROOT}")
target_include_directories(onnxruntime INTERFACE "${CMAKE_CURRENT_LIST_DIR}/external/onnxruntime/include")
# --------------------------------------------------------------

# ----------- My library and executable -------------------------
add_library(et_runner STATIC 
  ${CMAKE_CURRENT_SOURCE_DIR}/src/et_runner.cpp
  ${ET_ROOT}/extension/data_loader/file_data_loader.cpp
  ${ET_ROOT}/extension/evalue_util/print_evalue.cpp
  ${ET_ROOT}/extension/runner_util/inputs.cpp
  ${ET_ROOT}/extension/runner_util/inputs_portable.cpp
  ${ET_ROOT}/runtime/executor/test/test_backend_compiler_lib.cpp
)
target_link_libraries(et_runner PUBLIC 
  cxx_setup 
  executorch 
  xnnpack_backend 
  extension_runner_util
  extension_module_static
  optimized_native_cpu_ops_lib
)

add_executable(onnx_test ${CMAKE_CURRENT_SOURCE_DIR}/src/onnx_test.cpp)
target_include_directories(onnx_test PRIVATE "{CMAKE_CURRENT_SOURCE_DIR}/external/onnxruntime/include")
target_link_libraries(onnx_test PUBLIC cxx_setup onnxruntime)

add_executable(test ${CMAKE_CURRENT_SOURCE_DIR}/src/test.cpp)
target_link_libraries(test PRIVATE cxx_setup et_runner)
# --------------------------------------------------------------


