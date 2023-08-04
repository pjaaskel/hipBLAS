
cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

project(chipStar-hipBLAS)
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/install" CACHE PATH "..." FORCE)
endif()

# print out where hipBLAS will be installed
message(STATUS "hipBLAS will be installed to ${CMAKE_INSTALL_PREFIX}")

# Check if H4I-HipBLAS and H4I-MKLShim directories exist, if not, execute "git submodule update --init"
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/H4I-HipBLAS" OR NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/H4I-MKLShim")
    message(STATUS "H4I-HipBLAS and H4I-MKLShim submodules not found. Initializing...")
    execute_process(COMMAND git submodule update --init
                    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                    RESULT_VARIABLE submodule_result)
    if(submodule_result)
        message(FATAL_ERROR "Failed to initialize submodules.")
    endif()
endif()

add_subdirectory(H4I-MKLShim)
add_subdirectory(H4I-HipBLAS)
