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
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
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

project(cpp-inference VERSION 0.1 LANGUAGES CXX)

# -------- Build Executorch -----------------------------------
# set(PYTHON_EXECUTABLE python)
# option(EXECUTORCH_BUILD_XNNPACK "" ON) # Build with Xnnpack backend
# option(EXECUTORCH_BUILD_EXTENSION_DATA_LOADER "" ON)
# option(EXECUTORCH_BUILD_EXTENSION_MODULE "" ON)
# option(EXECUTORCH_BUILD_OPTIMIZED "" ON)
# set(EXECUTORCH_ENABLE_LOGGING 1)

# set(ET_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/external/executorch)
# set(ET_BUILD_PATH ${CMAKE_BINARY_DIR}/external/executorch)

# add_subdirectory(
#     ${ET_ROOT}
#     ${ET_BUILD_PATH}
# )
# --------------------------------------------------------------

# ---- For some reason extensions must be found ----------------
# link_directories("${ET_ROOT}/cmake-out/extension/runner_util")
# find_library(UTIL_LIB extension_runner_util HINTS "${ET_ROOT}/cmake-out/extension/runner_util")
# if(NOT UTIL_LIB)
#   message(FATAL_ERROR "runner_util library not found")
# endif()
# --------------------------------------------------------------

# ----------- Find onnxruntime ---------------------------------
add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES 
  IMPORTED_LOCATION "${CMAKE_CURRENT_LIST_DIR}/external/onnxruntime/lib/libonnxruntime.so")
target_include_directories(onnxruntime INTERFACE "${CMAKE_CURRENT_LIST_DIR}/external/onnxruntime/include")
# --------------------------------------------------------------

# ----------- My library and executable -------------------------
add_executable(onnx_test ${CMAKE_CURRENT_SOURCE_DIR}/src/onnx_test.cpp)
target_link_libraries(onnx_test PUBLIC cxx_setup onnxruntime)

# add_executable(tf_test ${CMAKE_CURRENT_SOURCE_DIR}/src/tf_test.cpp)
# target_link_libraries(tf_test PUBLIC cxx_setup tensorflow-lite)

# add_executable(et_test 
#   ${CMAKE_CURRENT_SOURCE_DIR}/src/et_test.cpp
#   ${ET_ROOT}/extension/module/module.cpp
#   ${ET_ROOT}/extension/evalue_util/print_evalue.cpp
#   ${ET_ROOT}/extension/data_loader/file_data_loader.cpp
#   ${ET_ROOT}/extension/runner_util/inputs.cpp
#   ${ET_ROOT}/extension/runner_util/inputs_portable.cpp
#   ${ET_ROOT}/runtime/executor/test/test_backend_compiler_lib.cpp
# )
# target_link_libraries(et_test PRIVATE
#   cxx_setup
#   executorch
#   xnnpack_backend
#   extension_module_static
#   optimized_native_cpu_ops_lib
#   extension_runner_util
# )
# --------------------------------------------------------------


