#!/bin/bash

module purge
module use /home/pvelesko/local/modulefiles
module use /home/bertoni/projects/p075.hipblas/testing/chip-spv/modulefiles/
module load chip-spv
module load cmake

module load intel/oneapi/release/2023.0.0
module load mkl/latest
module load compiler/latest

module load intel_compute_runtime

rm -rf build
mkdir build
cd build
cmake -DCMAKE_CXX_COMPILER=clang++ -DUSE_ONEAPI=ON -DCMAKE_C_COMPILER=clang -DCMAKE_BUILD_TYPE=Debug -DBUILD_CLIENTS_SAMPLES=ON  -DCMAKE_INSTALL_PREFIX=$PWD/packages ..

make VERBOSE=1
module load spack thapi
iprof ./clients/staging/hipblas-example-sscal-d 
#make install
#module load gdb 
#clang++ -g --hip-path=/home/bertoni/projects/p075.hipblas/testing/chip-spv/build/install "CMakeFiles/hipblas-example-sscal.dir/example_sscal.cpp.o" "CMakeFiles/hipblas-example-sscal.dir/__/common/utility.cpp.o" -o ../staging/hipblas-example-sscal-d  -Wl,-rpath,/home/bertoni/projects/p075.hipblas/testing/hipblas/build/library/src:/home/bertoni/projects/p075.hipblas/testing/hipblas/build/library/src/oneApi_detail/deps: ../../library/src/libhipblas-d.so ../../library/src/oneApi_detail/deps/libsycl_wrapper.so -lsycl -Wl,-rpath=/soft/compilers/oneapi-2023.0.0/mkl/2023.0.0/lib/intel64 -lm -ldl -lmkl_sycl /soft/compilers/oneapi-2023.0.0/mkl/2023.0.0/lib/intel64/libmkl_intel_ilp64.so /soft/compilers/oneapi-2023.0.0/mkl/2023.0.0/lib/intel64/libmkl_sequential.so /soft/compilers/oneapi-2023.0.0/mkl/2023.0.0/lib/intel64/libmkl_core.so /home/bertoni/projects/p075.hipblas/testing/chip-spv/build/install/lib/libCHIP.so /soft/restricted/CNDA/emb/intel-gpu-umd/20221012.1-pvc-prq-29/driver/lib64/libze_loader.so /soft/libraries/khronos/loader/master-2022.05.18/lib64/libOpenCL.so -lpthread -lstdc++fs 
#gdb /home/bertoni/projects/p075.hipblas/testing/hipblas/build/clients/staging/hipblas-example-sscal-d 
