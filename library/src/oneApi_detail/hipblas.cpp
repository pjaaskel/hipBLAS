//#include <hip/hip_runtime.h>
#include <iostream>
#include "deps/onemkl.h"
#include <algorithm>

#include <functional>
#include <hip/hip_interop.h>
#include <hipblas.h>
#include <exceptions.hpp>
//#include <math.h>

#include "sycl_w.h"

// local functions
static hipblasStatus_t updateSyclHandlesToCrrStream(hipStream_t stream, syclblasHandle_t handle)
{
    // Obtain the handles to the LZ handlers.
    unsigned long lzHandles[4];
    int           nHandles = 4;
    hipGetBackendNativeHandles((uintptr_t)stream, lzHandles, &nHandles);

    auto backendName = hipGetBackendName();

    //Fix-Me : Should Sycl know hipStream_t??
    syclblas_set_stream(handle, lzHandles, nHandles, stream, backendName);
    return HIPBLAS_STATUS_SUCCESS;
}

// hipblas APIs
hipblasStatus_t hipblasCreate(hipblasHandle_t* handle)
try
{
    // create syclBlas
    syclblas_create((syclblasHandle_t*)handle);

    hipStream_t nullStream = NULL; // default or null stream
    // set stream to default NULL stream
    auto status = updateSyclHandlesToCrrStream(nullStream, (syclblasHandle_t)*handle);
    return status;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDestroy(hipblasHandle_t handle)
try
{
    return syclblas_destroy((syclblasHandle_t)handle);
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasSetStream(hipblasHandle_t handle, hipStream_t stream)
try
{
    return updateSyclHandlesToCrrStream(stream, (syclblasHandle_t)handle);
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasGetStream(hipblasHandle_t handle, hipStream_t* pStream)
try
{
    if(handle == nullptr)
    {
        return HIPBLAS_STATUS_NOT_INITIALIZED;
    }
    return syclblas_get_hipstream((syclblasHandle_t)handle, pStream);
}
catch(...)
{
    return exception_to_hipblas_status();
}

// atomics mode - cannot find corresponding atomics mode in oneMKL, default to ALLOWED
hipblasStatus_t hipblasGetAtomicsMode(hipblasHandle_t handle, hipblasAtomicsMode_t* atomics_mode)
try
{
    *atomics_mode = HIPBLAS_ATOMICS_ALLOWED;
    return HIPBLAS_STATUS_SUCCESS;
 }
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasSetAtomicsMode(hipblasHandle_t handle, hipblasAtomicsMode_t atomics_mode)
try
{
    // No op
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasSetVector(int n, int elemSize, const void* x, int incx, void* y, int incy)
try
{
    if (n == 0) {
        // nothing to copy hence return early
        return HIPBLAS_STATUS_SUCCESS;
    }
    // error handling
    if (n < 0 || incx <= 0 || incy <= 0 || elemSize <= 0 || !x || !y) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    if (incx == 1 && incy == 1) {
        // contiguous memory
        auto status = hipMemcpy(y, x, elemSize * n, hipMemcpyHostToDevice);
        return HIPBLAS_STATUS_SUCCESS;
    } else {
        // As of now we don't have any way to handle non-contiguous memory hence returning as not supported
        return HIPBLAS_STATUS_NOT_SUPPORTED;
    }
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasGetVector(int n, int elemSize, const void* x, int incx, void* y, int incy)
try
{
    if (n == 0) {
        // nothing to copy hence return early
        return HIPBLAS_STATUS_SUCCESS;
    }
    // error handling
    if (n < 0 || incx <= 0 || incy <= 0 || elemSize <= 0 || !x || !y) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    if (incx == 1 && incy == 1) {
        // contiguous memory
        auto status = hipMemcpy(y, x, elemSize * n, hipMemcpyDeviceToHost);
        return HIPBLAS_STATUS_SUCCESS;
    } else {
        // As of now we don't have any way to handle non-contiguous memory hence returning as not supported
        return HIPBLAS_STATUS_NOT_SUPPORTED;
    }
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t
    hipblasSetMatrix(int rows, int cols, int elemSize, const void* A, int lda, void* B, int ldb)
try
{
    if (rows == 0 || cols == 0) {
        return HIPBLAS_STATUS_SUCCESS;
    }
    if (rows<0 || cols<0 || elemSize <= 0 || lda <= 0 || ldb <= 0 ||
        rows > lda || rows > ldb || A == nullptr || B == nullptr) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    // contiguous h2d copy
    if(lda == rows && ldb == rows) {
        // static cast to avoid overflow
        auto no_of_bytes = static_cast<size_t>(elemSize) * static_cast<size_t>(rows)
                           * static_cast<size_t>(cols);
        auto status = hipMemcpy(B, A, no_of_bytes, hipMemcpyHostToDevice);
        return HIPBLAS_STATUS_SUCCESS;
    } else {
        // non-contiguous memory, don't have better handling yet
        return HIPBLAS_STATUS_NOT_INITIALIZED;
    }
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t
    hipblasGetMatrix(int rows, int cols, int elemSize, const void* A, int lda, void* B, int ldb)
try
{
    if (rows == 0 || cols == 0) {
        return HIPBLAS_STATUS_SUCCESS;
    }
    if (rows<0 || cols<0 || elemSize <= 0 || lda <= 0 || ldb <= 0 ||
        rows > lda || rows > ldb || A == nullptr || B == nullptr) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    // contiguous d2h copy
    if(lda == rows && ldb == rows) {
        // static cast to avoid overflow
        auto no_of_bytes = static_cast<size_t>(elemSize) * static_cast<size_t>(rows)
                           * static_cast<size_t>(cols);
        auto status = hipMemcpy(B, A, no_of_bytes, hipMemcpyDeviceToHost);
        return HIPBLAS_STATUS_SUCCESS;
    } else {
        // non-contiguous memory, don't have better handling yet
        return HIPBLAS_STATUS_NOT_INITIALIZED;
    }
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasSetVectorAsync(
    int n, int elemSize, const void* x, int incx, void* y, int incy, hipStream_t stream)
try
{
    if (n == 0) {
        // nothing to copy hence return early
        return HIPBLAS_STATUS_SUCCESS;
    }
    // error handling
    if (n < 0 || incx <= 0 || incy <= 0 || elemSize <= 0 || !x || !y) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    if (incx == 1 && incy == 1) {
        // contiguous memory
        auto status = hipMemcpyAsync(y, x, elemSize * n, hipMemcpyHostToDevice, stream);
        return HIPBLAS_STATUS_SUCCESS;
    } else {
        // As of now we don't have any way to handle non-contiguous memory hence returning as not supported
        return HIPBLAS_STATUS_NOT_SUPPORTED;
    }
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasGetVectorAsync(
    int n, int elemSize, const void* x, int incx, void* y, int incy, hipStream_t stream)
try
{
    if (n == 0) {
        // nothing to copy hence return early
        return HIPBLAS_STATUS_SUCCESS;
    }
    // error handling
    if (n < 0 || incx <= 0 || incy <= 0 || elemSize <= 0 || !x || !y) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    if (incx == 1 && incy == 1) {
        // contiguous memory
        auto status = hipMemcpyAsync(y, x, elemSize * n, hipMemcpyDeviceToHost, stream);
        return HIPBLAS_STATUS_SUCCESS;
    } else {
        // As of now we don't have any way to handle non-contiguous memory hence returning as not supported
        return HIPBLAS_STATUS_NOT_SUPPORTED;
    }
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasSetMatrixAsync(
    int rows, int cols, int elemSize, const void* A, int lda, void* B, int ldb, hipStream_t stream)
try
{
    if (rows == 0 || cols == 0) {
        return HIPBLAS_STATUS_SUCCESS;
    }
    if (rows<0 || cols<0 || elemSize <= 0 || lda <= 0 || ldb <= 0 ||
        rows > lda || rows > ldb || A == nullptr || B == nullptr) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    // contiguous h2d copy
    if(lda == rows && ldb == rows) {
        // static cast to avoid overflow
        auto no_of_bytes = static_cast<size_t>(elemSize) * static_cast<size_t>(rows)
                           * static_cast<size_t>(cols);
        auto status = hipMemcpyAsync(B, A, no_of_bytes, hipMemcpyHostToDevice, stream);
        return HIPBLAS_STATUS_SUCCESS;
    } else {
        // non-contiguous memory, don't have better handling yet
        return HIPBLAS_STATUS_NOT_INITIALIZED;
    }
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasGetMatrixAsync(
    int rows, int cols, int elemSize, const void* A, int lda, void* B, int ldb, hipStream_t stream)
try
{
    if (rows == 0 || cols == 0) {
        return HIPBLAS_STATUS_SUCCESS;
    }
    if (rows<0 || cols<0 || elemSize <= 0 || lda <= 0 || ldb <= 0 ||
        rows > lda || rows > ldb || A == nullptr || B == nullptr) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    // contiguous d2h copy
    if(lda == rows && ldb == rows) {
        // static cast to avoid overflow
        auto no_of_bytes = static_cast<size_t>(elemSize) * static_cast<size_t>(rows)
                           * static_cast<size_t>(cols);
        auto status = hipMemcpyAsync(B, A, no_of_bytes, hipMemcpyDeviceToHost, stream);
        return HIPBLAS_STATUS_SUCCESS;
    } else {
        // non-contiguous memory, don't have better handling yet
        return HIPBLAS_STATUS_NOT_INITIALIZED;
    }
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasGetInt8Datatype(hipblasHandle_t handle, hipblasInt8Datatype_t * int8Type)
try
{
    *int8Type = HIPBLAS_INT8_DATATYPE_INT8;
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasSetInt8Datatype(hipblasHandle_t handle, hipblasInt8Datatype_t int8Type)
try
{
    // No op
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

bool isDevicePointer(const void* ptr) {
    hipPointerAttribute_t attribs;
    hipError_t hip_status = hipPointerGetAttributes(&attribs, ptr);
    bool is_result_dev_ptr = true;
    if (attribs.memoryType != hipMemoryTypeDevice) {
        is_result_dev_ptr = false;
    }
    return is_result_dev_ptr;
}

onemklTranspose convert(hipblasOperation_t val) {
    switch(val) {
        case HIPBLAS_OP_T:
            return ONEMKL_TRANSPOSE_TRANS;
        case HIPBLAS_OP_C:
            return ONEMLK_TRANSPOSE_CONJTRANS;
        case HIPBLAS_OP_N:
        default:
            return ONEMKL_TRANSPOSE_NONTRANS;
    }
}

onemklUplo convert(hipblasFillMode_t val) {
    switch(val) {
        case HIPBLAS_FILL_MODE_UPPER:
            return ONEMKL_UPLO_UPPER;
        case HIPBLAS_FILL_MODE_LOWER:
            return ONEMKL_UPLO_LOWER;
    }
}

onemklDiag convert(hipblasDiagType_t val) {
    switch(val) {
        case HIPBLAS_DIAG_NON_UNIT:
            return ONEMKL_DIAG_NONUNIT;
        case HIPBLAS_DIAG_UNIT:
            return ONEMKL_DIAG_UNIT;
    }
}

onemklSideMode convert(hipblasSideMode_t val) {
    switch(val) {
        case HIPBLAS_SIDE_LEFT:
            return ONEMKL_SIDE_LEFT;
        case HIPBLAS_SIDE_RIGHT:
            return ONEMKL_SIDE_RIGHT;
    }
}

hipblasPointerMode_t queryCurrentPtrMode(hipblasHandle_t handle) {
    int currMode;
    syclGetPointerMode((syclblasHandle_t)handle, &currMode);
    return (hipblasPointerMode_t)currMode;
}

// ----------------------------- hipBlas APIs ------------------------------------

hipblasStatus_t
hipblasSetPointerMode(hipblasHandle_t handle, hipblasPointerMode_t mode)
try
{
    if (handle == nullptr) {
        return HIPBLAS_STATUS_NOT_INITIALIZED;
    }
    if (mode != HIPBLAS_POINTER_MODE_HOST && mode != HIPBLAS_POINTER_MODE_DEVICE) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    return syclSetPointerMode((syclblasHandle_t)handle, (int)mode);
}
catch (...) {
    return exception_to_hipblas_status();
}

hipblasStatus_t
hipblasGetPointerMode(hipblasHandle_t handle, hipblasPointerMode_t* mode)
try
{
    if (mode == nullptr) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    if (handle == nullptr) {
        return HIPBLAS_STATUS_NOT_INITIALIZED;
    }
    int currPointerMode;
    auto status = syclGetPointerMode((syclblasHandle_t)handle, &currPointerMode);

    *mode = static_cast<hipblasPointerMode_t>(currPointerMode);
    return status;
}
catch (...) {
    return exception_to_hipblas_status();
}

// Level-1 : amax (supported datatypes : float, double, complex float, complex double)
// Generic amax which can handle batched/stride/non-batched
hipblasStatus_t hipblasIsamax(hipblasHandle_t handle, int n, const float* x, int incx, int* result)
try
{
    if (x == nullptr) {
        if (queryCurrentPtrMode(handle) == HIPBLAS_POINTER_MODE_DEVICE){
            hipMemset(result, 0, sizeof(int));
        } else {
            *result = 0;
        }
        return HIPBLAS_STATUS_SUCCESS;
    }
    // error checks
    if (handle == nullptr || result == nullptr || incx <= 0 || n <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }

    hipError_t hip_status;
    bool is_result_dev_ptr = (queryCurrentPtrMode(handle) == HIPBLAS_POINTER_MODE_DEVICE);
    int64_t *dev_results = nullptr;
    hip_status = hipMalloc(&dev_results, sizeof(int64_t));

    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    onemklSamax(sycl_queue, n, x, incx, dev_results);
    syclblas_queue_wait(sycl_queue); // wait until task is completed

    // Workaround:
    // 1. 'return' index in case of oneMKL starts from 0 to (n-1) but others return it as 1 to n hence
    // handling it below by incrementing it by one
    // 2.'result' in hipBLAS is 'int' but oneMKL accepts int64_t hence copying in stagging buffer and then copying it back to result
    int64_t results_host_memory = 0;
    hip_status = hipMemcpy(&results_host_memory, dev_results, sizeof(int64_t), hipMemcpyDefault);
    results_host_memory += 1;

    int return_val = (int)results_host_memory;
    if (is_result_dev_ptr) {
        hipMemcpy(result, &return_val, sizeof(int), hipMemcpyDefault);
    } else {
        *result = return_val;
    }

    hip_status = hipFree(&dev_results);
    return HIPBLAS_STATUS_SUCCESS;
}
catch (...) {
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasIdamax(hipblasHandle_t handle, int n, const double* x, int incx, int* result)
try
{
    if (x == nullptr) {
        if (queryCurrentPtrMode(handle) == HIPBLAS_POINTER_MODE_DEVICE){
            hipMemset(result, 0, sizeof(int));
        } else {
            *result = 0;
        }
        return HIPBLAS_STATUS_SUCCESS;
    }
    // error checks
    if (handle == nullptr || result == nullptr || incx <= 0 || n <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }

    hipError_t hip_status;
    bool is_result_dev_ptr = (queryCurrentPtrMode(handle) == HIPBLAS_POINTER_MODE_DEVICE);
    int64_t *dev_results = nullptr;
    hip_status = hipMalloc(&dev_results, sizeof(int64_t));

    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    onemklDamax(sycl_queue, n, x, incx, dev_results);
    syclblas_queue_wait(sycl_queue); // wait until task is completed

    // Workaround:
    // 1. 'return' index in case of oneMKL starts from 0 to (n-1) but others return it as 1 to n hence
    // handling it below by incrementing it by one
    // 2.'result' in hipBLAS is 'int' but oneMKL accepts int64_t hence copying in stagging buffer and then copying it back to result
    int64_t results_host_memory = 0;
    hip_status = hipMemcpy(&results_host_memory, dev_results, sizeof(int64_t), hipMemcpyDefault);
    results_host_memory += 1;

    int return_val = (int)results_host_memory;
    if (is_result_dev_ptr) {
        hipMemcpy(result, &return_val, sizeof(int), hipMemcpyDefault);
    } else {
        *result = return_val;
    }
    hip_status = hipFree(&dev_results);

    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasIcamax(hipblasHandle_t handle, int n, const hipblasComplex* x, int incx, int* result)
try
{
    if (x == nullptr) {
        if (queryCurrentPtrMode(handle) == HIPBLAS_POINTER_MODE_DEVICE){
            hipMemset(result, 0, sizeof(int));
        } else {
            *result = 0;
        }
        return HIPBLAS_STATUS_SUCCESS;
    }
    // error checks
    if (handle == nullptr || result == nullptr || incx <= 0 || n <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }

    hipError_t hip_status;
    bool is_result_dev_ptr = (queryCurrentPtrMode(handle) == HIPBLAS_POINTER_MODE_DEVICE);
    int64_t *dev_results = nullptr;
    hip_status = hipMalloc(&dev_results, sizeof(int64_t));

    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    onemklCamax(sycl_queue, n, (const float _Complex*)x, incx, dev_results);

    syclblas_queue_wait(sycl_queue); // wait until task is completed

    // Workaround:
    // 1. 'return' index in case of oneMKL starts from 0 to (n-1) but others return it as 1 to n hence
    // handling it below by incrementing it by one
    // 2.'result' in hipBLAS is 'int' but oneMKL accepts int64_t hence copying in stagging buffer and then copying it back to result
    int64_t results_host_memory = 0;
    hip_status = hipMemcpy(&results_host_memory, dev_results, sizeof(int64_t), hipMemcpyDefault);
    results_host_memory += 1;

    int return_val = (int)results_host_memory;
    if (is_result_dev_ptr) {
        hipMemcpy(result, &return_val, sizeof(int), hipMemcpyDefault);
    } else {
        *result = return_val;
    }
    hip_status = hipFree(&dev_results);

    return HIPBLAS_STATUS_SUCCESS;
}
catch (...) {
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasIzamax(hipblasHandle_t handle, int n, const hipblasDoubleComplex* x, int incx, int* result)
try
{
    if (x == nullptr) {
        if (queryCurrentPtrMode(handle) == HIPBLAS_POINTER_MODE_DEVICE){
            int a = 0;
            hipMemcpy(result, &a, sizeof(int), hipMemcpyDefault);
        } else {
            *result = 0;
        }
        return HIPBLAS_STATUS_SUCCESS;
    }
    // error checks
    if (handle == nullptr || result == nullptr || incx <= 0 || n <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }

    hipError_t hip_status;
    bool is_result_dev_ptr = (queryCurrentPtrMode(handle) == HIPBLAS_POINTER_MODE_DEVICE);
    int64_t *dev_results = nullptr;
    hip_status = hipMalloc(&dev_results, sizeof(int64_t));

    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    onemklZamax(sycl_queue, n, (const double _Complex*)x, incx, dev_results);
    syclblas_queue_wait(sycl_queue); // wait until task is completed

    // Workaround:
    // 1. 'return' index in case of oneMKL starts from 0 to (n-1) but others return it as 1 to n hence
    // handling it below by incrementing it by one
    // 2.'result' in hipBLAS is 'int' but oneMKL accepts int64_t hence copying in stagging buffer
    //   then copying it back to result
    int64_t results_host_memory = 0;
    hip_status = hipMemcpy(&results_host_memory, dev_results, sizeof(int64_t), hipMemcpyDefault);
    results_host_memory += 1;

    int return_val = (int)results_host_memory;
    if (is_result_dev_ptr) {
        hipMemcpy(result, &return_val, sizeof(int), hipMemcpyDefault);
    } else {
        *result = return_val;
    }
    hip_status = hipFree(&dev_results);

    return HIPBLAS_STATUS_SUCCESS;
}
catch (...) {
    return exception_to_hipblas_status();
}
// amax_batched
hipblasStatus_t hipblasIsamaxBatched(
    hipblasHandle_t handle, int n, const float* const x[], int incx, int batchCount, int* result)
try
{
return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasIdamaxBatched(
    hipblasHandle_t handle, int n, const double* const x[], int incx, int batchCount, int* result)
try
{
return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasIcamaxBatched(hipblasHandle_t             handle,
                                     int                         n,
                                     const hipblasComplex* const x[],
                                     int                         incx,
                                     int                         batchCount,
                                     int*                        result)
try
{
return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasIzamaxBatched(hipblasHandle_t                   handle,
                                     int                               n,
                                     const hipblasDoubleComplex* const x[],
                                     int                               incx,
                                     int                               batchCount,
                                     int*                              result)
try
{
return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

// amax_strided_batched
hipblasStatus_t hipblasIsamaxStridedBatched(hipblasHandle_t handle,
                                            int             n,
                                            const float*    x,
                                            int             incx,
                                            hipblasStride   stridex,
                                            int             batchCount,
                                            int*            result)
try
{
return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasIdamaxStridedBatched(hipblasHandle_t handle,
                                            int             n,
                                            const double*   x,
                                            int             incx,
                                            hipblasStride   stridex,
                                            int             batchCount,
                                            int*            result)
try
{
return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasIcamaxStridedBatched(hipblasHandle_t       handle,
                                            int                   n,
                                            const hipblasComplex* x,
                                            int                   incx,
                                            hipblasStride         stridex,
                                            int                   batchCount,
                                            int*                  result)
try
{
return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasIzamaxStridedBatched(hipblasHandle_t             handle,
                                            int                         n,
                                            const hipblasDoubleComplex* x,
                                            int                         incx,
                                            hipblasStride               stridex,
                                            int                         batchCount,
                                            int*                        result)
try
{
return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

// Level-1 : amin (supported datatypes : float, double, complex float, complex double)
// Generic amin which can handle batched/stride/non-batched
hipblasStatus_t hipblasIsamin(hipblasHandle_t handle, int n, const float* x, int incx, int* result)
try
{
    if (x == nullptr) {
        if (queryCurrentPtrMode(handle) == HIPBLAS_POINTER_MODE_DEVICE){
            hipMemset(result, 0, sizeof(int));
        } else {
            *result = 0;
        }
        return HIPBLAS_STATUS_SUCCESS;
    }
    // error checks
    if (handle == nullptr || result == nullptr || incx <= 0 || n <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }

    hipError_t hip_status;
    bool is_result_dev_ptr = (queryCurrentPtrMode(handle) == HIPBLAS_POINTER_MODE_DEVICE);
    int64_t *dev_results = nullptr;
    hip_status = hipMalloc(&dev_results, sizeof(int64_t));

    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    onemklSamin(sycl_queue, n, x, incx, dev_results);
    syclblas_queue_wait(sycl_queue); // wait until task is completed

    // Workaround:
    // 1. 'return' index in case of oneMKL starts from 0 to (n-1) but others return it as 1 to n hence
    // handling it below by incrementing it by one
    // 2.'result' in hipBLAS is 'int' but oneMKL accepts int64_t hence copying in stagging buffer and then copying it back to result
    int64_t results_host_memory = 0;
    hip_status = hipMemcpy(&results_host_memory, dev_results, sizeof(int64_t), hipMemcpyDefault);
    results_host_memory += 1;

    int return_val = (int)results_host_memory;
    if (is_result_dev_ptr) {
        hipMemcpy(result, &return_val, sizeof(int), hipMemcpyDefault);
    } else {
        *result = return_val;
    }
    hip_status = hipFree(&dev_results);
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasIdamin(hipblasHandle_t handle, int n, const double* x, int incx, int* result)
try
{
    if (x == nullptr) {
        if (queryCurrentPtrMode(handle) == HIPBLAS_POINTER_MODE_DEVICE){
            hipMemset(result, 0, sizeof(int));
        } else {
            *result = 0;
        }
        return HIPBLAS_STATUS_SUCCESS;
    }
    // error checks
    if (handle == nullptr || result == nullptr || incx <= 0 || n <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }

    hipError_t hip_status;
    bool is_result_dev_ptr = (queryCurrentPtrMode(handle) == HIPBLAS_POINTER_MODE_DEVICE);
    int64_t *dev_results = nullptr;
    hip_status = hipMalloc(&dev_results, sizeof(int64_t));

    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    onemklDamin(sycl_queue, n, x, incx, dev_results);
    syclblas_queue_wait(sycl_queue); // wait until task is completed

    // Workaround:
    // 1. 'return' index in case of oneMKL starts from 0 to (n-1) but others return it as 1 to n hence
    // handling it below by incrementing it by one
    // 2.'result' in hipBLAS is 'int' but oneMKL accepts int64_t hence copying in stagging buffer and then copying it back to result
    int64_t results_host_memory = 0;
    hip_status = hipMemcpy(&results_host_memory, dev_results, sizeof(int64_t), hipMemcpyDefault);
    results_host_memory += 1;

    int return_val = (int)results_host_memory;
    if (is_result_dev_ptr) {
        hipMemcpy(result, &return_val, sizeof(int), hipMemcpyDefault);
    } else {
        *result = return_val;
    }
    hip_status = hipFree(&dev_results);
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasIcamin(hipblasHandle_t handle, int n, const hipblasComplex* x, int incx, int* result)
try
{
    if (x == nullptr) {
        if (queryCurrentPtrMode(handle) == HIPBLAS_POINTER_MODE_DEVICE){
            hipMemset(result, 0, sizeof(int));
        } else {
            *result = 0;
        }
        return HIPBLAS_STATUS_SUCCESS;
    }
    // error checks
    if (handle == nullptr || result == nullptr || incx <= 0 || n <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }

    hipError_t hip_status;
    bool is_result_dev_ptr = (queryCurrentPtrMode(handle) == HIPBLAS_POINTER_MODE_DEVICE);
    int64_t *dev_results = nullptr;
    hip_status = hipMalloc(&dev_results, sizeof(int64_t));

    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    onemklCamin(sycl_queue, n, (const float _Complex*)x, incx, dev_results);

    syclblas_queue_wait(sycl_queue); // wait until task is completed

    // Workaround:
    // 1. 'return' index in case of oneMKL starts from 0 to (n-1) but others return it as 1 to n hence
    // handling it below by incrementing it by one
    // 2.'result' in hipBLAS is 'int' but oneMKL accepts int64_t hence copying in stagging buffer and then copying it back to result
    int64_t results_host_memory = 0;
    hip_status = hipMemcpy(&results_host_memory, dev_results, sizeof(int64_t), hipMemcpyDefault);
    results_host_memory += 1;

    int return_val = (int)results_host_memory;
    if (is_result_dev_ptr) {
        hipMemcpy(result, &return_val, sizeof(int), hipMemcpyDefault);
    } else {
        *result = return_val;
    }
    hip_status = hipFree(&dev_results);
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasIzamin(hipblasHandle_t handle, int n, const hipblasDoubleComplex* x, int incx, int* result)
try
{
    if (x == nullptr) {
        if (queryCurrentPtrMode(handle) == HIPBLAS_POINTER_MODE_DEVICE){
            hipMemset(result, 0, sizeof(int));
        } else {
            *result = 0;
        }
        return HIPBLAS_STATUS_SUCCESS;
    }
    // error checks
    if (handle == nullptr || result == nullptr || incx <= 0 || n <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }

    hipError_t hip_status;
    bool is_result_dev_ptr = (queryCurrentPtrMode(handle) == HIPBLAS_POINTER_MODE_DEVICE);
    int64_t *dev_results = nullptr;
    hip_status = hipMalloc(&dev_results, sizeof(int64_t));

    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    onemklZamin(sycl_queue, n, (const double _Complex*)x, incx, dev_results);

    syclblas_queue_wait(sycl_queue); // wait until task is completed

    // Workaround:
    // 1. 'return' index in case of oneMKL starts from 0 to (n-1) but others return it as 1 to n hence
    // handling it below by incrementing it by one
    // 2.'result' in hipBLAS is 'int' but oneMKL accepts int64_t hence copying in stagging buffer and then copying it back to result
    int64_t results_host_memory = 0;
    hip_status = hipMemcpy(&results_host_memory, dev_results, sizeof(int64_t), hipMemcpyDefault);
    results_host_memory += 1;

    int return_val = (int)results_host_memory;
    if (is_result_dev_ptr) {
        hipMemcpy(result, &return_val, sizeof(int), hipMemcpyDefault);
    } else {
        *result = return_val;
    }
    hip_status = hipFree(&dev_results);
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}
// amin_batched
hipblasStatus_t hipblasIsaminBatched(
    hipblasHandle_t handle, int n, const float* const x[], int incx, int batchCount, int* result)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasIdaminBatched(
    hipblasHandle_t handle, int n, const double* const x[], int incx, int batchCount, int* result)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasIcaminBatched(hipblasHandle_t             handle,
                                     int                         n,
                                     const hipblasComplex* const x[],
                                     int                         incx,
                                     int                         batchCount,
                                     int*                        result)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasIzaminBatched(hipblasHandle_t                   handle,
                                     int                               n,
                                     const hipblasDoubleComplex* const x[],
                                     int                               incx,
                                     int                               batchCount,
                                     int*                              result)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

// amin_strided_batched
hipblasStatus_t hipblasIsaminStridedBatched(hipblasHandle_t handle,
                                            int             n,
                                            const float*    x,
                                            int             incx,
                                            hipblasStride   stridex,
                                            int             batchCount,
                                            int*            result)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasIdaminStridedBatched(hipblasHandle_t handle,
                                            int             n,
                                            const double*   x,
                                            int             incx,
                                            hipblasStride   stridex,
                                            int             batchCount,
                                            int*            result)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasIcaminStridedBatched(hipblasHandle_t       handle,
                                            int                   n,
                                            const hipblasComplex* x,
                                            int                   incx,
                                            hipblasStride         stridex,
                                            int                   batchCount,
                                            int*                  result)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasIzaminStridedBatched(hipblasHandle_t             handle,
                                            int                         n,
                                            const hipblasDoubleComplex* x,
                                            int                         incx,
                                            hipblasStride               stridex,
                                            int                         batchCount,
                                            int*                        result)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

// Level-1 : asum (supported datatypes : float, double, complex float, complex double)
// Generic asum which can handle batched/stride/non-batched
hipblasStatus_t hipblasSasum(hipblasHandle_t handle, int n, const float* x, int incx, float* result)
try
{
    // error checks
    if (handle == nullptr || x == nullptr || result == nullptr || incx <= 0 || n <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    hipError_t hip_status;
    bool is_result_dev_ptr = isDevicePointer(result);
    // 'result' can be device or host memory but oneMKL needs device memory
    float* dev_result = result;
    if (!is_result_dev_ptr) {
        hip_status = hipMalloc(&dev_result, sizeof(float));
    }

    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    onemklSasum(sycl_queue, n, x, incx, dev_result);
    syclblas_queue_wait(sycl_queue);

    if (!is_result_dev_ptr) {
        hip_status = hipMemcpy(result, dev_result, sizeof(float), hipMemcpyDefault);
        hip_status = hipFree(dev_result);
    }
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t
    hipblasDasum(hipblasHandle_t handle, int n, const double* x, int incx, double* result)
try
{
    // error checks
    if (handle == nullptr || x == nullptr || result == nullptr || incx <= 0 || n <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    hipError_t hip_status;
    bool is_result_dev_ptr = isDevicePointer(result);
    // 'result' can be device or host memory but oneMKL needs device memory
    double* dev_result = result;
    if (!is_result_dev_ptr) {
        hip_status = hipMalloc(&dev_result, sizeof(double));
    }

    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    onemklDasum(sycl_queue, n, x, incx, dev_result);
    syclblas_queue_wait(sycl_queue);

    if (!is_result_dev_ptr) {
        hip_status = hipMemcpy(result, dev_result, sizeof(double), hipMemcpyDefault);
        hip_status = hipFree(dev_result);
    }

    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasScasum(hipblasHandle_t handle, int n, const hipblasComplex* x, int incx, float* result)
try
{
    // error checks
    if (handle == nullptr || x == nullptr || result == nullptr || incx <= 0 || n <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    hipError_t hip_status;
    bool is_result_dev_ptr = isDevicePointer(result);
    // 'result' can be device or host memory but oneMKL needs device memory
    float* dev_result = result;
    if (!is_result_dev_ptr) {
        hip_status = hipMalloc(&dev_result, sizeof(float));
    }

    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    onemklCasum(sycl_queue, n, (const float _Complex*)x, incx, dev_result);
    syclblas_queue_wait(sycl_queue);

    if (!is_result_dev_ptr) {
        hip_status = hipMemcpy(result, dev_result, sizeof(float), hipMemcpyDefault);
        hip_status = hipFree(dev_result);
    }

    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDzasum(hipblasHandle_t handle, int n, const hipblasDoubleComplex* x, int incx, double* result)
try
{
    // error checks
    if (handle == nullptr || x == nullptr || result == nullptr || incx <= 0 || n <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    hipError_t hip_status;
    bool is_result_dev_ptr = isDevicePointer(result);
    // 'result' can be device or host memory but oneMKL needs device memory
    double* dev_result = result;
    if (!is_result_dev_ptr) {
        hip_status = hipMalloc(&dev_result, sizeof(double));
    }

    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    onemklZasum(sycl_queue, n, (const double _Complex*)x, incx, dev_result);
    syclblas_queue_wait(sycl_queue);

    if (!is_result_dev_ptr) {
        hip_status = hipMemcpy(result, dev_result, sizeof(double), hipMemcpyDefault);
        hip_status = hipFree(dev_result);
    }
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}
// asum_batched
hipblasStatus_t hipblasSasumBatched(
    hipblasHandle_t handle, int n, const float* const x[], int incx, int batchCount, float* result)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDasumBatched(hipblasHandle_t     handle,
                                    int                 n,
                                    const double* const x[],
                                    int                 incx,
                                    int                 batchCount,
                                    double*             result)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasScasumBatched(hipblasHandle_t             handle,
                                     int                         n,
                                     const hipblasComplex* const x[],
                                     int                         incx,
                                     int                         batchCount,
                                     float*                      result)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDzasumBatched(hipblasHandle_t                   handle,
                                     int                               n,
                                     const hipblasDoubleComplex* const x[],
                                     int                               incx,
                                     int                               batchCount,
                                     double*                           result)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

// asum_strided_batched
hipblasStatus_t hipblasSasumStridedBatched(hipblasHandle_t handle,
                                           int             n,
                                           const float*    x,
                                           int             incx,
                                           hipblasStride   stridex,
                                           int             batchCount,
                                           float*          result)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDasumStridedBatched(hipblasHandle_t handle,
                                           int             n,
                                           const double*   x,
                                           int             incx,
                                           hipblasStride   stridex,
                                           int             batchCount,
                                           double*         result)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasScasumStridedBatched(hipblasHandle_t       handle,
                                            int                   n,
                                            const hipblasComplex* x,
                                            int                   incx,
                                            hipblasStride         stridex,
                                            int                   batchCount,
                                            float*                result)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDzasumStridedBatched(hipblasHandle_t             handle,
                                            int                         n,
                                            const hipblasDoubleComplex* x,
                                            int                         incx,
                                            hipblasStride               stridex,
                                            int                         batchCount,
                                            double*                     result)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

// Level-1 : axpy (supported datatypes : float, double, complex float, complex double)
// Generic axpy which can handle batched/stride/non-batched
hipblasStatus_t hipblasHaxpy(hipblasHandle_t    handle,
                             int                n,
                             const hipblasHalf* alpha,
                             const hipblasHalf* x,
                             int                incx,
                             hipblasHalf*       y,
                             int                incy)
try
{
    // error checks
    if (handle == nullptr || x == nullptr || y == nullptr ||alpha == nullptr ||
        n <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    // 'alpha' can be device or host memory hence need to be copied before access
    uint16_t host_alpha_ptr = 0;

    if (queryCurrentPtrMode(handle) == HIPBLAS_POINTER_MODE_DEVICE) {
        auto hipStatus = hipMemcpy(&host_alpha_ptr, alpha, sizeof(uint16_t), hipMemcpyDefault);
    } else {
        host_alpha_ptr = *alpha;
    }

    onemklHaxpy(sycl_queue, n, host_alpha_ptr, (const short*)x, incx, (short*)y, incy);
    syclblas_queue_wait(sycl_queue);

    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasSaxpy(hipblasHandle_t handle, int n, const float* alpha,
                             const float* x, int incx, float* y, int incy)
try
{
    // error checks
    if (handle == nullptr || x == nullptr || y == nullptr ||alpha == nullptr ||
        n <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    // 'alpha' can be device or host memory hence need to be copied before access
    float host_alpha_ptr = 0;
    
    if (queryCurrentPtrMode(handle) == HIPBLAS_POINTER_MODE_DEVICE) {
        auto hipStatus = hipMemcpy(&host_alpha_ptr, alpha, sizeof(float), hipMemcpyDefault);
    } else {
        host_alpha_ptr = *alpha;
    }

    onemklSaxpy(sycl_queue, n, host_alpha_ptr, x, incx, y, incy);
    syclblas_queue_wait(sycl_queue);

    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDaxpy(hipblasHandle_t handle, int n, const double* alpha,
                             const double* x, int incx, double* y, int incy)
try
{
    // error checks
    if (handle == nullptr || x == nullptr || y == nullptr ||alpha == nullptr ||
        n <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    // 'alpha' can be device or host memory hence need to be copied before access
    double host_alpha_ptr = 0;
    if (queryCurrentPtrMode(handle)  == HIPBLAS_POINTER_MODE_DEVICE) {
        auto hipStatus = hipMemcpy(&host_alpha_ptr, alpha, sizeof(double), hipMemcpyDefault);
    } else {
        host_alpha_ptr = *alpha;
    }

    onemklDaxpy(sycl_queue, n, host_alpha_ptr, x, incx, y, incy);

    syclblas_queue_wait(sycl_queue);

    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCaxpy(hipblasHandle_t handle, int n, const hipblasComplex* alpha,
                             const hipblasComplex* x, int incx, hipblasComplex* y, int incy)
try
{
    // error checks
    if (handle == nullptr || x == nullptr || y == nullptr ||alpha == nullptr ||
        n <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    // 'alpha' can be device or host memory hence need to be copied before access
    float _Complex host_alpha_ptr = 0;
    if (queryCurrentPtrMode(handle) == HIPBLAS_POINTER_MODE_DEVICE) {
        auto hipStatus = hipMemcpy(&host_alpha_ptr, alpha, sizeof(float _Complex), hipMemcpyDefault);
    } else {
        host_alpha_ptr = *((const float _Complex*)alpha);
    }
    onemklCaxpy(sycl_queue, n, host_alpha_ptr, (const float _Complex*)x, incx, (float _Complex*)y, incy);

    syclblas_queue_wait(sycl_queue);

    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZaxpy(hipblasHandle_t handle, int n, const hipblasDoubleComplex* alpha,
                             const hipblasDoubleComplex* x, int incx, hipblasDoubleComplex* y, int incy)
try
{
    // error checks
    if (handle == nullptr || x == nullptr || y == nullptr ||alpha == nullptr ||
        n <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    // 'alpha' can be device or host memory hence need to be copied before access
    double _Complex host_alpha_ptr = 0;
    if (queryCurrentPtrMode(handle) == HIPBLAS_POINTER_MODE_DEVICE) {
        auto hipStatus = hipMemcpy(&host_alpha_ptr, alpha, sizeof(double _Complex), hipMemcpyDefault);
    } else {
        host_alpha_ptr = *((const double _Complex*)alpha);
    }

    onemklZaxpy(sycl_queue, n, host_alpha_ptr, (const double _Complex*)x, incx, (double _Complex*)y, incy);

    syclblas_queue_wait(sycl_queue);

    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}
// axpy_batched
hipblasStatus_t hipblasHaxpyBatched(hipblasHandle_t          handle,
                                    int                      n,
                                    const hipblasHalf*       alpha,
                                    const hipblasHalf* const x[],
                                    int                      incx,
                                    hipblasHalf* const       y[],
                                    int                      incy,
                                    int                      batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasSaxpyBatched(hipblasHandle_t    handle,
                                    int                n,
                                    const float*       alpha,
                                    const float* const x[],
                                    int                incx,
                                    float* const       y[],
                                    int                incy,
                                    int                batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDaxpyBatched(hipblasHandle_t     handle,
                                    int                 n,
                                    const double*       alpha,
                                    const double* const x[],
                                    int                 incx,
                                    double* const       y[],
                                    int                 incy,
                                    int                 batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCaxpyBatched(hipblasHandle_t             handle,
                                    int                         n,
                                    const hipblasComplex*       alpha,
                                    const hipblasComplex* const x[],
                                    int                         incx,
                                    hipblasComplex* const       y[],
                                    int                         incy,
                                    int                         batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZaxpyBatched(hipblasHandle_t                   handle,
                                    int                               n,
                                    const hipblasDoubleComplex*       alpha,
                                    const hipblasDoubleComplex* const x[],
                                    int                               incx,
                                    hipblasDoubleComplex* const       y[],
                                    int                               incy,
                                    int                               batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

// axpy_strided_batched
hipblasStatus_t hipblasHaxpyStridedBatched(hipblasHandle_t    handle,
                                           int                n,
                                           const hipblasHalf* alpha,
                                           const hipblasHalf* x,
                                           int                incx,
                                           hipblasStride      stridex,
                                           hipblasHalf*       y,
                                           int                incy,
                                           hipblasStride      stridey,
                                           int                batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasSaxpyStridedBatched(hipblasHandle_t handle,
                                           int             n,
                                           const float*    alpha,
                                           const float*    x,
                                           int             incx,
                                           hipblasStride   stridex,
                                           float*          y,
                                           int             incy,
                                           hipblasStride   stridey,
                                           int             batchCount)
try
{
    // error checks
    if (handle == nullptr || x == nullptr || y == nullptr ||alpha == nullptr ||
        n <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    // 'alpha' can be device or host memory hence need to be copied before access
    float host_alpha_ptr = 0;

    if (queryCurrentPtrMode(handle) == HIPBLAS_POINTER_MODE_DEVICE) {
        auto hipStatus = hipMemcpy(&host_alpha_ptr, alpha, sizeof(float), hipMemcpyDefault);
    } else {
        host_alpha_ptr = *alpha;
    }

    onemklSaxpy_strided(sycl_queue, n, host_alpha_ptr, x, incx, stridex, y, incy, stridey, batchCount);
    syclblas_queue_wait(sycl_queue);

    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDaxpyStridedBatched(hipblasHandle_t handle,
                                           int             n,
                                           const double*   alpha,
                                           const double*   x,
                                           int             incx,
                                           hipblasStride   stridex,
                                           double*         y,
                                           int             incy,
                                           hipblasStride   stridey,
                                           int             batchCount)
try
{
    // error checks
    if (handle == nullptr || x == nullptr || y == nullptr ||alpha == nullptr ||
        n <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    // 'alpha' can be device or host memory hence need to be copied before access
    double host_alpha_ptr = 0;
    if (queryCurrentPtrMode(handle)  == HIPBLAS_POINTER_MODE_DEVICE) {
        auto hipStatus = hipMemcpy(&host_alpha_ptr, alpha, sizeof(double), hipMemcpyDefault);
    } else {
        host_alpha_ptr = *alpha;
    }

    onemklDaxpy_strided(sycl_queue, n, host_alpha_ptr, x, incx, stridex, y, incy, stridey, batchCount);

    syclblas_queue_wait(sycl_queue);

    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCaxpyStridedBatched(hipblasHandle_t       handle,
                                           int                   n,
                                           const hipblasComplex* alpha,
                                           const hipblasComplex* x,
                                           int                   incx,
                                           hipblasStride         stridex,
                                           hipblasComplex*       y,
                                           int                   incy,
                                           hipblasStride         stridey,
                                           int                   batchCount)
try
{
    // error checks
    if (handle == nullptr || x == nullptr || y == nullptr ||alpha == nullptr ||
        n <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    // 'alpha' can be device or host memory hence need to be copied before access
    float _Complex host_alpha_ptr = 0;
    if (queryCurrentPtrMode(handle) == HIPBLAS_POINTER_MODE_DEVICE) {
        auto hipStatus = hipMemcpy(&host_alpha_ptr, alpha, sizeof(float _Complex), hipMemcpyDefault);
    } else {
        host_alpha_ptr = *((const float _Complex*)alpha);
    }
    onemklCaxpy_strided(sycl_queue, n, host_alpha_ptr, (const float _Complex*)x, incx, stridex, (float _Complex*)y, incy, stridey, batchCount);

    syclblas_queue_wait(sycl_queue);

    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZaxpyStridedBatched(hipblasHandle_t             handle,
                                           int                         n,
                                           const hipblasDoubleComplex* alpha,
                                           const hipblasDoubleComplex* x,
                                           int                         incx,
                                           hipblasStride               stridex,
                                           hipblasDoubleComplex*       y,
                                           int                         incy,
                                           hipblasStride               stridey,
                                           int                         batchCount)
try
{
    // error checks
    if (handle == nullptr || x == nullptr || y == nullptr ||alpha == nullptr ||
        n <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    // 'alpha' can be device or host memory hence need to be copied before access
    double _Complex host_alpha_ptr = 0;
    if (queryCurrentPtrMode(handle) == HIPBLAS_POINTER_MODE_DEVICE) {
        auto hipStatus = hipMemcpy(&host_alpha_ptr, alpha, sizeof(double _Complex), hipMemcpyDefault);
    } else {
        host_alpha_ptr = *((const double _Complex*)alpha);
    }

    onemklZaxpy_strided(sycl_queue, n, host_alpha_ptr, (const double _Complex*)x, incx, stridex,
                        (double _Complex*)y, incy, stridey, batchCount);

    syclblas_queue_wait(sycl_queue);

    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

// Level-1 : copy (supported datatypes : float, double, complex float, complex double)
// Generic copy which can handle batched/stride/non-batched
hipblasStatus_t
    hipblasScopy(hipblasHandle_t handle, int n, const float* x, int incx, float* y, int incy)
try
{
    // error checks
    if (handle == nullptr || x == nullptr || y == nullptr ||
        incx <= 0 || incy <= 0 || n <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    onemklScopy(sycl_queue, n, x, incx, y, incy);
    syclblas_queue_wait(sycl_queue);
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t
    hipblasDcopy(hipblasHandle_t handle, int n, const double* x, int incx, double* y, int incy)
try
{
    // error checks
    if (handle == nullptr || x == nullptr || y == nullptr ||
        incx <= 0 || incy <= 0 || n <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    onemklDcopy(sycl_queue, n, x, incx, y, incy);
    syclblas_queue_wait(sycl_queue);
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t
    hipblasCcopy(hipblasHandle_t handle, int n, const hipblasComplex* x, int incx, hipblasComplex* y, int incy)
try
{
    // error checks
    if (handle == nullptr || x == nullptr || y == nullptr ||
        incx <= 0 || incy <= 0 || n <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    onemklCcopy(sycl_queue, n, (const float _Complex*)x, incx, (float _Complex*)y, incy);
    syclblas_queue_wait(sycl_queue);
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t
    hipblasZcopy(hipblasHandle_t handle, int n, const hipblasDoubleComplex* x, int incx, hipblasDoubleComplex* y, int incy)
try
{
    // error checks
    if (handle == nullptr || x == nullptr || y == nullptr ||
        incx <= 0 || incy <= 0 || n <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    onemklZcopy(sycl_queue, n, (const double _Complex*)x, incx, (double _Complex*)y, incy);
    syclblas_queue_wait(sycl_queue);
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}
// copy_batched
hipblasStatus_t hipblasScopyBatched(hipblasHandle_t    handle,
                                    int                n,
                                    const float* const x[],
                                    int                incx,
                                    float* const       y[],
                                    int                incy,
                                    int                batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDcopyBatched(hipblasHandle_t     handle,
                                    int                 n,
                                    const double* const x[],
                                    int                 incx,
                                    double* const       y[],
                                    int                 incy,
                                    int                 batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCcopyBatched(hipblasHandle_t             handle,
                                    int                         n,
                                    const hipblasComplex* const x[],
                                    int                         incx,
                                    hipblasComplex* const       y[],
                                    int                         incy,
                                    int                         batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZcopyBatched(hipblasHandle_t                   handle,
                                    int                               n,
                                    const hipblasDoubleComplex* const x[],
                                    int                               incx,
                                    hipblasDoubleComplex* const       y[],
                                    int                               incy,
                                    int                               batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

// copy_strided_batched
hipblasStatus_t hipblasScopyStridedBatched(hipblasHandle_t handle,
                                           int             n,
                                           const float*    x,
                                           int             incx,
                                           hipblasStride   stridex,
                                           float*          y,
                                           int             incy,
                                           hipblasStride   stridey,
                                           int             batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDcopyStridedBatched(hipblasHandle_t handle,
                                           int             n,
                                           const double*   x,
                                           int             incx,
                                           hipblasStride   stridex,
                                           double*         y,
                                           int             incy,
                                           hipblasStride   stridey,
                                           int             batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCcopyStridedBatched(hipblasHandle_t       handle,
                                           int                   n,
                                           const hipblasComplex* x,
                                           int                   incx,
                                           hipblasStride         stridex,
                                           hipblasComplex*       y,
                                           int                   incy,
                                           hipblasStride         stridey,
                                           int                   batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZcopyStridedBatched(hipblasHandle_t             handle,
                                           int                         n,
                                           const hipblasDoubleComplex* x,
                                           int                         incx,
                                           hipblasStride               stridex,
                                           hipblasDoubleComplex*       y,
                                           int                         incy,
                                           hipblasStride               stridey,
                                           int                         batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

// Level-1 : dot (supported datatypes : float, double, complex float, complex double)
// Generic dot which can handle batched/stride/non-batched
hipblasStatus_t hipblasHdot(hipblasHandle_t handle, int n, const hipblasHalf* x, int incx,
                            const hipblasHalf* y, int incy, hipblasHalf* result)
try
{
    // error checks
    if (handle == nullptr || x == nullptr || y == nullptr ||result == nullptr ||
        incx <= 0 || incy <= 0 || n <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    hipError_t hip_status;
    bool is_result_dev_ptr = (queryCurrentPtrMode(handle) == HIPBLAS_POINTER_MODE_DEVICE);
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    short* dev_result;
    dev_result = (short *)result;
    if (!is_result_dev_ptr) {
        hip_status = hipMalloc(&dev_result, sizeof(short));
    }
    onemklHdot(sycl_queue, n, (const short*)x, incx, (const short *)y, incy, dev_result);
    syclblas_queue_wait(sycl_queue);

    if (!is_result_dev_ptr) {
        hip_status = hipMemcpy(result, dev_result, sizeof(uint16_t), hipMemcpyDefault);
        hip_status = hipFree(dev_result);
    }

    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasSdot(hipblasHandle_t handle, int n, const float* x, int incx, const float* y, int incy, float* result)
try
{
    // error checks
    if (handle == nullptr || x == nullptr || y == nullptr ||result == nullptr ||
        incx <= 0 || incy <= 0 || n <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }    
    hipError_t hip_status;
    bool is_result_dev_ptr = (queryCurrentPtrMode(handle) == HIPBLAS_POINTER_MODE_DEVICE);
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    float* dev_result = result;
    if (!is_result_dev_ptr) {
        hip_status = hipMalloc(&dev_result, sizeof(float));
    }
    onemklSdot(sycl_queue, n, x, incx, y, incy, dev_result);
    syclblas_queue_wait(sycl_queue);

    if (!is_result_dev_ptr) {
        hip_status = hipMemcpy(result, dev_result, sizeof(float), hipMemcpyDefault);
        hip_status = hipFree(dev_result);
    }

    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDdot(hipblasHandle_t handle, int n, const double* x, int incx, const double* y, int incy, double* result)
try
{
    // error checks
    if (handle == nullptr || x == nullptr || y == nullptr ||result == nullptr ||
        incx <= 0 || incy <= 0 || n <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }    
    hipError_t hip_status;
    bool is_result_dev_ptr = (queryCurrentPtrMode(handle) == HIPBLAS_POINTER_MODE_DEVICE);
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    double* dev_result = result;
    if (!is_result_dev_ptr) {
        hip_status = hipMalloc(&dev_result, sizeof(double));
    }
    onemklDdot(sycl_queue, n, x, incx, y, incy, dev_result);
    syclblas_queue_wait(sycl_queue);

    if (!is_result_dev_ptr) {
        hip_status = hipMemcpy(result, dev_result, sizeof(double), hipMemcpyDefault);
        hip_status = hipFree(dev_result);
    }

    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCdotc(hipblasHandle_t handle, int n, const hipblasComplex* x,
                             int incx, const hipblasComplex* y, int incy, hipblasComplex* result)
try
{
    // error checks
    if (handle == nullptr || x == nullptr || y == nullptr ||result == nullptr ||
        incx <= 0 || incy <= 0 || n <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }    
    hipError_t hip_status;
    bool is_result_dev_ptr = (queryCurrentPtrMode(handle) == HIPBLAS_POINTER_MODE_DEVICE);
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    float _Complex* dev_result = (float _Complex*)result;
    if (!is_result_dev_ptr) {
        hip_status = hipMalloc(&dev_result, sizeof(float _Complex));
    }
    onemklCdotc(sycl_queue, n, (const float _Complex*)x, incx, (const float _Complex*)y, incy, dev_result);
    syclblas_queue_wait(sycl_queue);

    if (!is_result_dev_ptr) {
        hip_status = hipMemcpy(result, dev_result, sizeof(float _Complex), hipMemcpyDefault);
        hip_status = hipFree(dev_result);
    }

    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCdotu(hipblasHandle_t handle, int n, const hipblasComplex* x,
                             int incx, const hipblasComplex* y, int incy, hipblasComplex* result)
try
{
    // error checks
    if (handle == nullptr || x == nullptr || y == nullptr ||result == nullptr ||
        incx <= 0 || incy <= 0 || n <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }    
    hipError_t hip_status;
    bool is_result_dev_ptr = (queryCurrentPtrMode(handle) == HIPBLAS_POINTER_MODE_DEVICE);
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    float _Complex* dev_result = (float _Complex*)result;
    if (!is_result_dev_ptr) {
        hip_status = hipMalloc(&dev_result, sizeof(float _Complex));
    }
    onemklCdotu(sycl_queue, n, (const float _Complex*)x, incx, (const float _Complex*)y, incy, dev_result);
    syclblas_queue_wait(sycl_queue);

    if (!is_result_dev_ptr) {
        hip_status = hipMemcpy(result, dev_result, sizeof(float _Complex), hipMemcpyDefault);
        hip_status = hipFree(dev_result);
    }

    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZdotc(hipblasHandle_t handle, int n, const hipblasDoubleComplex* x,
                             int incx, const hipblasDoubleComplex* y, int incy, hipblasDoubleComplex* result)
try
{
    // error checks
    if (handle == nullptr || x == nullptr || y == nullptr ||result == nullptr ||
        incx <= 0 || incy <= 0 || n <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }    
    hipError_t hip_status;
    bool is_result_dev_ptr = (queryCurrentPtrMode(handle) == HIPBLAS_POINTER_MODE_DEVICE);
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    double _Complex* dev_result = (double _Complex*)result;
    if (!is_result_dev_ptr) {
        hip_status = hipMalloc(&dev_result, sizeof(double _Complex));
    }
    onemklZdotc(sycl_queue, n, (const double _Complex*)x, incx, (const double _Complex*)y, incy, dev_result);
    syclblas_queue_wait(sycl_queue);

    if (!is_result_dev_ptr) {
        hip_status = hipMemcpy(result, dev_result, sizeof(double _Complex), hipMemcpyDefault);
        hip_status = hipFree(dev_result);
    }
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZdotu(hipblasHandle_t handle, int n, const hipblasDoubleComplex* x,
                             int incx, const hipblasDoubleComplex* y, int incy, hipblasDoubleComplex* result)
try
{
    // error checks
    if (handle == nullptr || x == nullptr || y == nullptr ||result == nullptr ||
        incx <= 0 || incy <= 0 || n <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }    
    hipError_t hip_status;
    bool is_result_dev_ptr = (queryCurrentPtrMode(handle) == HIPBLAS_POINTER_MODE_DEVICE);
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    double _Complex* dev_result = (double _Complex*)result;
    if (!is_result_dev_ptr) {
        hip_status = hipMalloc(&dev_result, sizeof(double _Complex));
    }
    onemklZdotu(sycl_queue, n, (const double _Complex*)x, incx, (const double _Complex*)y, incy, dev_result);
    syclblas_queue_wait(sycl_queue);

    if (!is_result_dev_ptr) {
        hip_status = hipMemcpy(result, dev_result, sizeof(double _Complex), hipMemcpyDefault);
        hip_status = hipFree(dev_result);
    }
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}
// dot_batched
hipblasStatus_t hipblasHdotBatched(hipblasHandle_t          handle,
                                   int                      n,
                                   const hipblasHalf* const x[],
                                   int                      incx,
                                   const hipblasHalf* const y[],
                                   int                      incy,
                                   int                      batchCount,
                                   hipblasHalf*             result)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasBfdotBatched(hipblasHandle_t              handle,
                                    int                          n,
                                    const hipblasBfloat16* const x[],
                                    int                          incx,
                                    const hipblasBfloat16* const y[],
                                    int                          incy,
                                    int                          batchCount,
                                    hipblasBfloat16*             result)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasSdotBatched(hipblasHandle_t    handle,
                                   int                n,
                                   const float* const x[],
                                   int                incx,
                                   const float* const y[],
                                   int                incy,
                                   int                batchCount,
                                   float*             result)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDdotBatched(hipblasHandle_t     handle,
                                   int                 n,
                                   const double* const x[],
                                   int                 incx,
                                   const double* const y[],
                                   int                 incy,
                                   int                 batchCount,
                                   double*             result)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCdotcBatched(hipblasHandle_t             handle,
                                    int                         n,
                                    const hipblasComplex* const x[],
                                    int                         incx,
                                    const hipblasComplex* const y[],
                                    int                         incy,
                                    int                         batchCount,
                                    hipblasComplex*             result)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCdotuBatched(hipblasHandle_t             handle,
                                    int                         n,
                                    const hipblasComplex* const x[],
                                    int                         incx,
                                    const hipblasComplex* const y[],
                                    int                         incy,
                                    int                         batchCount,
                                    hipblasComplex*             result)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZdotcBatched(hipblasHandle_t                   handle,
                                    int                               n,
                                    const hipblasDoubleComplex* const x[],
                                    int                               incx,
                                    const hipblasDoubleComplex* const y[],
                                    int                               incy,
                                    int                               batchCount,
                                    hipblasDoubleComplex*             result)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZdotuBatched(hipblasHandle_t                   handle,
                                    int                               n,
                                    const hipblasDoubleComplex* const x[],
                                    int                               incx,
                                    const hipblasDoubleComplex* const y[],
                                    int                               incy,
                                    int                               batchCount,
                                    hipblasDoubleComplex*             result)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

// dot_strided_batched
hipblasStatus_t hipblasHdotStridedBatched(hipblasHandle_t    handle,
                                          int                n,
                                          const hipblasHalf* x,
                                          int                incx,
                                          hipblasStride      stridex,
                                          const hipblasHalf* y,
                                          int                incy,
                                          hipblasStride      stridey,
                                          int                batchCount,
                                          hipblasHalf*       result)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasBfdotStridedBatched(hipblasHandle_t        handle,
                                           int                    n,
                                           const hipblasBfloat16* x,
                                           int                    incx,
                                           hipblasStride          stridex,
                                           const hipblasBfloat16* y,
                                           int                    incy,
                                           hipblasStride          stridey,
                                           int                    batchCount,
                                           hipblasBfloat16*       result)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasSdotStridedBatched(hipblasHandle_t handle,
                                          int             n,
                                          const float*    x,
                                          int             incx,
                                          hipblasStride   stridex,
                                          const float*    y,
                                          int             incy,
                                          hipblasStride   stridey,
                                          int             batchCount,
                                          float*          result)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDdotStridedBatched(hipblasHandle_t handle,
                                          int             n,
                                          const double*   x,
                                          int             incx,
                                          hipblasStride   stridex,
                                          const double*   y,
                                          int             incy,
                                          hipblasStride   stridey,
                                          int             batchCount,
                                          double*         result)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCdotcStridedBatched(hipblasHandle_t       handle,
                                           int                   n,
                                           const hipblasComplex* x,
                                           int                   incx,
                                           hipblasStride         stridex,
                                           const hipblasComplex* y,
                                           int                   incy,
                                           hipblasStride         stridey,
                                           int                   batchCount,
                                           hipblasComplex*       result)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCdotuStridedBatched(hipblasHandle_t       handle,
                                           int                   n,
                                           const hipblasComplex* x,
                                           int                   incx,
                                           hipblasStride         stridex,
                                           const hipblasComplex* y,
                                           int                   incy,
                                           hipblasStride         stridey,
                                           int                   batchCount,
                                           hipblasComplex*       result)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZdotcStridedBatched(hipblasHandle_t             handle,
                                           int                         n,
                                           const hipblasDoubleComplex* x,
                                           int                         incx,
                                           hipblasStride               stridex,
                                           const hipblasDoubleComplex* y,
                                           int                         incy,
                                           hipblasStride               stridey,
                                           int                         batchCount,
                                           hipblasDoubleComplex*       result)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZdotuStridedBatched(hipblasHandle_t             handle,
                                           int                         n,
                                           const hipblasDoubleComplex* x,
                                           int                         incx,
                                           hipblasStride               stridex,
                                           const hipblasDoubleComplex* y,
                                           int                         incy,
                                           hipblasStride               stridey,
                                           int                         batchCount,
                                           hipblasDoubleComplex*       result)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

// Level-1 : nrm2 (supported datatypes : float, double, complex float, complex double)
hipblasStatus_t
    hipblasSnrm2(hipblasHandle_t handle, int n, const float* x, int incx, float* result)
try
{
    // error checks
    if (handle == nullptr || x == nullptr || result == nullptr ||
        incx <= 0 || n <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }    
    hipError_t status;
    bool is_result_dev_ptr = (queryCurrentPtrMode(handle) == HIPBLAS_POINTER_MODE_DEVICE);
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    float* dev_result = result;
    if (!is_result_dev_ptr) {
        status = hipMalloc(&dev_result, sizeof(float));
    }
    onemklSnrm2(sycl_queue, n, x, incx, dev_result);
    syclblas_queue_wait(sycl_queue);

    if (!is_result_dev_ptr) {
        status = hipMemcpy(result, dev_result, sizeof(float), hipMemcpyDefault);
        status = hipFree(dev_result);
    }
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t
    hipblasDnrm2(hipblasHandle_t handle, int n, const double* x, int incx, double* result)
try
{
    // error checks
    if (handle == nullptr || x == nullptr || result == nullptr ||
        incx <= 0 || n <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }    
    hipError_t status;
    bool is_result_dev_ptr = (queryCurrentPtrMode(handle) == HIPBLAS_POINTER_MODE_DEVICE);
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    double* dev_result = result;
    if (!is_result_dev_ptr) {
        status = hipMalloc(&dev_result, sizeof(double));
    }
    onemklDnrm2(sycl_queue, n, x, incx, dev_result);
    syclblas_queue_wait(sycl_queue);

    if (!is_result_dev_ptr) {
        status = hipMemcpy(result, dev_result, sizeof(double), hipMemcpyDefault);
        status = hipFree(dev_result);
    }
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t
    hipblasScnrm2(hipblasHandle_t handle, int n, const hipblasComplex* x, int incx, float* result)
try
{
    // error checks
    if (handle == nullptr || x == nullptr || result == nullptr ||
        incx <= 0 || n <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }    
    hipError_t status;
    bool is_result_dev_ptr = (queryCurrentPtrMode(handle) == HIPBLAS_POINTER_MODE_DEVICE);
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    float* dev_result = result;
    if (!is_result_dev_ptr) {
        status = hipMalloc(&dev_result, sizeof(float));
    }
    onemklCnrm2(sycl_queue, n, (const float _Complex*)x, incx, dev_result);
    syclblas_queue_wait(sycl_queue);
    if (!is_result_dev_ptr) {
        status = hipMemcpy(result, dev_result, sizeof(float), hipMemcpyDefault);
        status = hipFree(dev_result);
    }

    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t
    hipblasDznrm2(hipblasHandle_t handle, int n, const hipblasDoubleComplex* x, int incx, double* result)
try
{
    // error checks
    if (handle == nullptr || x == nullptr || result == nullptr ||
        incx <= 0 || n <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }    
    hipError_t status;
    bool is_result_dev_ptr = (queryCurrentPtrMode(handle) == HIPBLAS_POINTER_MODE_DEVICE);
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    double* dev_result = result;
    if (!is_result_dev_ptr) {
        status = hipMalloc(&dev_result, sizeof(double));
    }
    onemklZnrm2(sycl_queue, n, (const double _Complex*)x, incx, dev_result);
    syclblas_queue_wait(sycl_queue);
    if (!is_result_dev_ptr) {
        status = hipMemcpy(result, dev_result, sizeof(double), hipMemcpyDefault);
        status = hipFree(dev_result);
    }
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}
// nrm2_batched
hipblasStatus_t hipblasSnrm2Batched(
    hipblasHandle_t handle, int n, const float* const x[], int incx, int batchCount, float* result)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDnrm2Batched(hipblasHandle_t     handle,
                                    int                 n,
                                    const double* const x[],
                                    int                 incx,
                                    int                 batchCount,
                                    double*             result)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasScnrm2Batched(hipblasHandle_t             handle,
                                     int                         n,
                                     const hipblasComplex* const x[],
                                     int                         incx,
                                     int                         batchCount,
                                     float*                      result)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDznrm2Batched(hipblasHandle_t                   handle,
                                     int                               n,
                                     const hipblasDoubleComplex* const x[],
                                     int                               incx,
                                     int                               batchCount,
                                     double*                           result)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

// nrm2_strided_batched
hipblasStatus_t hipblasSnrm2StridedBatched(hipblasHandle_t handle,
                                           int             n,
                                           const float*    x,
                                           int             incx,
                                           hipblasStride   stridex,
                                           int             batchCount,
                                           float*          result)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDnrm2StridedBatched(hipblasHandle_t handle,
                                           int             n,
                                           const double*   x,
                                           int             incx,
                                           hipblasStride   stridex,
                                           int             batchCount,
                                           double*         result)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasScnrm2StridedBatched(hipblasHandle_t       handle,
                                            int                   n,
                                            const hipblasComplex* x,
                                            int                   incx,
                                            hipblasStride         stridex,
                                            int                   batchCount,
                                            float*                result)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDznrm2StridedBatched(hipblasHandle_t             handle,
                                            int                         n,
                                            const hipblasDoubleComplex* x,
                                            int                         incx,
                                            hipblasStride               stridex,
                                            int                         batchCount,
                                            double*                     result)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

// Level-1 : rot (supported datatypes : float, double, complex float, complex double)
hipblasStatus_t hipblasSrot(hipblasHandle_t handle,int n, float* x,int incx,
                                           float* y, int incy,const float* c, const float* s)
try
{
    // error checks
    if (handle == nullptr ) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    if (x == nullptr || y == nullptr || c == nullptr || s == nullptr) {
        return HIPBLAS_STATUS_SUCCESS;
    }
    hipError_t hip_status;
    bool is_ptr_mode_device = (queryCurrentPtrMode(handle) == HIPBLAS_POINTER_MODE_DEVICE);
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    float h_c, h_s;
    if (is_ptr_mode_device) {
        hip_status = hipMemcpy(&h_c, c, sizeof(float), hipMemcpyDefault);
        hip_status = hipMemcpy(&h_s, s, sizeof(float), hipMemcpyDefault);
    } else {
        h_c = *c;
        h_s = *s;
    }

    onemklSrot(sycl_queue, n, x, incx, y, incy, h_c, h_s);
    syclblas_queue_wait(sycl_queue);
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDrot(hipblasHandle_t handle,int n, double* x,int incx,
                                           double* y, int incy,const double* c, const double* s)
try
{
    // error checks
    if (handle == nullptr ) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    if (x == nullptr || y == nullptr || c == nullptr || s == nullptr) {
        return HIPBLAS_STATUS_SUCCESS;
    }
    hipError_t hip_status;
    bool is_ptr_mode_device = (queryCurrentPtrMode(handle) == HIPBLAS_POINTER_MODE_DEVICE);
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    double h_c, h_s;
    if (is_ptr_mode_device) {
        hip_status = hipMemcpy(&h_c, c, sizeof(double), hipMemcpyDefault);
        hip_status = hipMemcpy(&h_s, s, sizeof(double), hipMemcpyDefault);
    } else {
        h_c = *c;
        h_s = *s;
    }

    onemklDrot(sycl_queue, n, x, incx, y, incy, h_c, h_s);
    syclblas_queue_wait(sycl_queue);
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCrot(hipblasHandle_t handle,int n, hipblasComplex* x,int incx,
                                           hipblasComplex* y, int incy,const float* c,
                                           const hipblasComplex* s)
try
{
    // error checks
    if (handle == nullptr ) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    if (x == nullptr || y == nullptr || c == nullptr || s == nullptr) {
        return HIPBLAS_STATUS_SUCCESS;
    }

    hipError_t hip_status;
    bool is_ptr_mode_device = (queryCurrentPtrMode(handle) == HIPBLAS_POINTER_MODE_DEVICE);
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    float h_c;
    float _Complex h_s;
    if (is_ptr_mode_device) {
        hip_status = hipMemcpy(&h_c, c, sizeof(float), hipMemcpyDefault);
        hip_status = hipMemcpy(&h_s, s, sizeof(float _Complex), hipMemcpyDefault);
    } else {
        h_c = *c;
        h_s = *((float _Complex*)s);
    }

    onemklCrot(sycl_queue, n, (float _Complex*)x, incx, (float _Complex*)y, incy, h_c, h_s);
    syclblas_queue_wait(sycl_queue);
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCsrot(hipblasHandle_t handle,int n, hipblasComplex* x,int incx,
                                           hipblasComplex* y, int incy,const float* c,
                                           const float* s)
try
{
    // error checks
    if (handle == nullptr ) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    if (x == nullptr || y == nullptr || c == nullptr || s == nullptr) {
        return HIPBLAS_STATUS_SUCCESS;
    }

    hipError_t hip_status;
    bool is_ptr_mode_device = (queryCurrentPtrMode(handle) == HIPBLAS_POINTER_MODE_DEVICE);
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    float h_c, h_s;
    if (is_ptr_mode_device) {
        hip_status = hipMemcpy(&h_c, c, sizeof(float), hipMemcpyDefault);
        hip_status = hipMemcpy(&h_s, s, sizeof(float), hipMemcpyDefault);
    } else {
        h_c = *c;
        h_s = *s;
    }

    onemklCsrot(sycl_queue, n, (float _Complex*)x, incx, (float _Complex*)y, incy, h_c, h_s);
    syclblas_queue_wait(sycl_queue);
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZrot(hipblasHandle_t handle,int n, hipblasDoubleComplex* x,int incx,
                                           hipblasDoubleComplex* y, int incy,const double* c,
                                           const hipblasDoubleComplex* s)
try
{
    // error checks
    if (handle == nullptr ) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    if (x == nullptr || y == nullptr || c == nullptr || s == nullptr) {
        return HIPBLAS_STATUS_SUCCESS;
    }

    hipError_t hip_status;
    bool is_ptr_mode_device = (queryCurrentPtrMode(handle) == HIPBLAS_POINTER_MODE_DEVICE);
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    double h_c;
    double _Complex h_s;
    if (is_ptr_mode_device) {
        hip_status = hipMemcpy(&h_c, c, sizeof(double), hipMemcpyDefault);
        hip_status = hipMemcpy(&h_s, s, sizeof(double _Complex), hipMemcpyDefault);
    } else {
        h_c = *c;
        h_s = *((double _Complex*)s);
    }

    onemklZrot(sycl_queue, n, (double _Complex*)x, incx, (double _Complex*)y, incy, h_c, h_s);
    syclblas_queue_wait(sycl_queue);
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZdrot(hipblasHandle_t handle,int n, hipblasDoubleComplex* x,int incx,
                                           hipblasDoubleComplex* y, int incy, const double* c,
                                           const double* s)
try
{
    // error checks
    if (handle == nullptr ) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    if (x == nullptr || y == nullptr || c == nullptr || s == nullptr) {
        return HIPBLAS_STATUS_SUCCESS;
    }
    hipError_t hip_status;
    bool is_ptr_mode_device = (queryCurrentPtrMode(handle) == HIPBLAS_POINTER_MODE_DEVICE);
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    double h_c, h_s;
    if (is_ptr_mode_device) {
        hip_status = hipMemcpy(&h_c, c, sizeof(double), hipMemcpyDefault);
        hip_status = hipMemcpy(&h_s, s, sizeof(double), hipMemcpyDefault);
    } else {
        h_c = *c;
        h_s = *s;
    }

    onemklZdrot(sycl_queue, n, (double _Complex*)x, incx, (double _Complex*)y, incy, h_c, h_s);
    syclblas_queue_wait(sycl_queue);
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}
// rot_batched
hipblasStatus_t hipblasSrotBatched(hipblasHandle_t handle,
                                   int             n,
                                   float* const    x[],
                                   int             incx,
                                   float* const    y[],
                                   int             incy,
                                   const float*    c,
                                   const float*    s,
                                   int             batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDrotBatched(hipblasHandle_t handle,
                                   int             n,
                                   double* const   x[],
                                   int             incx,
                                   double* const   y[],
                                   int             incy,
                                   const double*   c,
                                   const double*   s,
                                   int             batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCrotBatched(hipblasHandle_t       handle,
                                   int                   n,
                                   hipblasComplex* const x[],
                                   int                   incx,
                                   hipblasComplex* const y[],
                                   int                   incy,
                                   const float*          c,
                                   const hipblasComplex* s,
                                   int                   batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCsrotBatched(hipblasHandle_t       handle,
                                    int                   n,
                                    hipblasComplex* const x[],
                                    int                   incx,
                                    hipblasComplex* const y[],
                                    int                   incy,
                                    const float*          c,
                                    const float*          s,
                                    int                   batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZrotBatched(hipblasHandle_t             handle,
                                   int                         n,
                                   hipblasDoubleComplex* const x[],
                                   int                         incx,
                                   hipblasDoubleComplex* const y[],
                                   int                         incy,
                                   const double*               c,
                                   const hipblasDoubleComplex* s,
                                   int                         batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZdrotBatched(hipblasHandle_t             handle,
                                    int                         n,
                                    hipblasDoubleComplex* const x[],
                                    int                         incx,
                                    hipblasDoubleComplex* const y[],
                                    int                         incy,
                                    const double*               c,
                                    const double*               s,
                                    int                         batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

// rot_strided_batched
hipblasStatus_t hipblasSrotStridedBatched(hipblasHandle_t handle,
                                          int             n,
                                          float*          x,
                                          int             incx,
                                          hipblasStride   stridex,
                                          float*          y,
                                          int             incy,
                                          hipblasStride   stridey,
                                          const float*    c,
                                          const float*    s,
                                          int             batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDrotStridedBatched(hipblasHandle_t handle,
                                          int             n,
                                          double*         x,
                                          int             incx,
                                          hipblasStride   stridex,
                                          double*         y,
                                          int             incy,
                                          hipblasStride   stridey,
                                          const double*   c,
                                          const double*   s,
                                          int             batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCrotStridedBatched(hipblasHandle_t       handle,
                                          int                   n,
                                          hipblasComplex*       x,
                                          int                   incx,
                                          hipblasStride         stridex,
                                          hipblasComplex*       y,
                                          int                   incy,
                                          hipblasStride         stridey,
                                          const float*          c,
                                          const hipblasComplex* s,
                                          int                   batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCsrotStridedBatched(hipblasHandle_t handle,
                                           int             n,
                                           hipblasComplex* x,
                                           int             incx,
                                           hipblasStride   stridex,
                                           hipblasComplex* y,
                                           int             incy,
                                           hipblasStride   stridey,
                                           const float*    c,
                                           const float*    s,
                                           int             batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZrotStridedBatched(hipblasHandle_t             handle,
                                          int                         n,
                                          hipblasDoubleComplex*       x,
                                          int                         incx,
                                          hipblasStride               stridex,
                                          hipblasDoubleComplex*       y,
                                          int                         incy,
                                          hipblasStride               stridey,
                                          const double*               c,
                                          const hipblasDoubleComplex* s,
                                          int                         batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZdrotStridedBatched(hipblasHandle_t       handle,
                                           int                   n,
                                           hipblasDoubleComplex* x,
                                           int                   incx,
                                           hipblasStride         stridex,
                                           hipblasDoubleComplex* y,
                                           int                   incy,
                                           hipblasStride         stridey,
                                           const double*         c,
                                           const double*         s,
                                           int                   batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

// Level-1 : rotg (supported datatypes : float, double, complex float, complex double)
hipblasStatus_t hipblasSrotg(hipblasHandle_t handle, float* a, float* b, float* c, float* s)
try
{
    if (handle == nullptr || a == nullptr || b == nullptr || c == nullptr || s == nullptr) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    // FixMe: oneAPI supports only device pointers
    if (queryCurrentPtrMode(handle) != HIPBLAS_POINTER_MODE_DEVICE) {
        return HIPBLAS_STATUS_NOT_SUPPORTED;
    }
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    onemklSrotg(sycl_queue, a, b, c, s);
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDrotg(hipblasHandle_t handle, double* a, double* b, double* c, double* s)
try
{
    if (handle == nullptr || a == nullptr || b == nullptr || c == nullptr || s == nullptr) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    // FixMe: oneAPI supports only device pointers
    if (queryCurrentPtrMode(handle) != HIPBLAS_POINTER_MODE_DEVICE) {
        return HIPBLAS_STATUS_NOT_SUPPORTED;
    }
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    onemklDrotg(sycl_queue, a, b, c, s);
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCrotg(hipblasHandle_t handle, hipblasComplex* a, hipblasComplex* b, float* c, hipblasComplex* s)
try
{
    if (handle == nullptr || a == nullptr || b == nullptr || c == nullptr || s == nullptr) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    // FixMe: oneAPI supports only device pointers
    if (queryCurrentPtrMode(handle) != HIPBLAS_POINTER_MODE_DEVICE) {
        return HIPBLAS_STATUS_NOT_SUPPORTED;
    }
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    onemklCrotg(sycl_queue, (float _Complex*)a, (float _Complex*)b, c, (float _Complex*)s);
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZrotg(hipblasHandle_t handle, hipblasDoubleComplex* a,
                             hipblasDoubleComplex* b, double* c, hipblasDoubleComplex* s)
try
{
    if (handle == nullptr || a == nullptr || b == nullptr || c == nullptr || s == nullptr) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    // FixMe: oneAPI supports only device pointers
    if (queryCurrentPtrMode(handle) != HIPBLAS_POINTER_MODE_DEVICE) {
        return HIPBLAS_STATUS_NOT_SUPPORTED;
    }
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    onemklZrotg(sycl_queue, (double _Complex*)a, (double _Complex*)b, c, (double _Complex*)s);
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}
// rotg_batched
hipblasStatus_t hipblasSrotgBatched(hipblasHandle_t handle,
                                    float* const    a[],
                                    float* const    b[],
                                    float* const    c[],
                                    float* const    s[],
                                    int             batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDrotgBatched(hipblasHandle_t handle,
                                    double* const   a[],
                                    double* const   b[],
                                    double* const   c[],
                                    double* const   s[],
                                    int             batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCrotgBatched(hipblasHandle_t       handle,
                                    hipblasComplex* const a[],
                                    hipblasComplex* const b[],
                                    float* const          c[],
                                    hipblasComplex* const s[],
                                    int                   batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZrotgBatched(hipblasHandle_t             handle,
                                    hipblasDoubleComplex* const a[],
                                    hipblasDoubleComplex* const b[],
                                    double* const               c[],
                                    hipblasDoubleComplex* const s[],
                                    int                         batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

// rotg_strided_batched
hipblasStatus_t hipblasSrotgStridedBatched(hipblasHandle_t handle,
                                           float*          a,
                                           hipblasStride   stride_a,
                                           float*          b,
                                           hipblasStride   stride_b,
                                           float*          c,
                                           hipblasStride   stride_c,
                                           float*          s,
                                           hipblasStride   stride_s,
                                           int             batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDrotgStridedBatched(hipblasHandle_t handle,
                                           double*         a,
                                           hipblasStride   stride_a,
                                           double*         b,
                                           hipblasStride   stride_b,
                                           double*         c,
                                           hipblasStride   stride_c,
                                           double*         s,
                                           hipblasStride   stride_s,
                                           int             batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCrotgStridedBatched(hipblasHandle_t handle,
                                           hipblasComplex* a,
                                           hipblasStride   stride_a,
                                           hipblasComplex* b,
                                           hipblasStride   stride_b,
                                           float*          c,
                                           hipblasStride   stride_c,
                                           hipblasComplex* s,
                                           hipblasStride   stride_s,
                                           int             batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZrotgStridedBatched(hipblasHandle_t       handle,
                                           hipblasDoubleComplex* a,
                                           hipblasStride         stride_a,
                                           hipblasDoubleComplex* b,
                                           hipblasStride         stride_b,
                                           double*               c,
                                           hipblasStride         stride_c,
                                           hipblasDoubleComplex* s,
                                           hipblasStride         stride_s,
                                           int                   batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

// Level-1 : rotm (supported datatypes : float, double)
hipblasStatus_t hipblasSrotm(hipblasHandle_t handle, int n, float* x, int incx, float* y, int incy, const float* param)
try
{
    // error checks
    if (handle == nullptr) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    if (n <= 0 ||x == nullptr || y == nullptr ||param == nullptr) {
        return HIPBLAS_STATUS_SUCCESS;
    }

    hipError_t hipStatus;
    bool is_param_dev_ptr = (queryCurrentPtrMode(handle) == HIPBLAS_POINTER_MODE_DEVICE);
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    float* dev_param = (float*) param;
    if (!is_param_dev_ptr) {
        hipStatus = hipMalloc(&dev_param, sizeof(float)*5);
        hipStatus = hipMemcpy(dev_param, param, sizeof(float)*5, hipMemcpyHostToDevice);
    }

    onemklSrotm(sycl_queue, n, x, incx, y, incy, dev_param);

    if (!is_param_dev_ptr) {
        syclblas_queue_wait(sycl_queue);
        hipStatus = hipFree(dev_param);
    }
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDrotm(hipblasHandle_t handle, int n, double* x, int incx, double* y, int incy, const double* param)
try
{
    // error checks
    if (handle == nullptr) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    if (n <= 0 ||x == nullptr || y == nullptr ||param == nullptr) {
        return HIPBLAS_STATUS_SUCCESS;
    }
    hipError_t hipStatus;
    bool is_param_dev_ptr = (queryCurrentPtrMode(handle) == HIPBLAS_POINTER_MODE_DEVICE);
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    double* dev_param = (double*)param;
    if (!is_param_dev_ptr) {
        hipStatus = hipMalloc(&dev_param, sizeof(double)*5);
        hipStatus = hipMemcpy(dev_param, param, sizeof(double)*5, hipMemcpyHostToDevice);
    }

    onemklDrotm(sycl_queue, n, x, incx, y, incy, dev_param);

    if (!is_param_dev_ptr) {
        syclblas_queue_wait(sycl_queue);
        hipStatus = hipFree(dev_param);
    }
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}
// rotm_batched
hipblasStatus_t hipblasSrotmBatched(hipblasHandle_t    handle,
                                    int                n,
                                    float* const       x[],
                                    int                incx,
                                    float* const       y[],
                                    int                incy,
                                    const float* const param[],
                                    int                batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDrotmBatched(hipblasHandle_t     handle,
                                    int                 n,
                                    double* const       x[],
                                    int                 incx,
                                    double* const       y[],
                                    int                 incy,
                                    const double* const param[],
                                    int                 batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

// rotm_strided_batched
hipblasStatus_t hipblasSrotmStridedBatched(hipblasHandle_t handle,
                                           int             n,
                                           float*          x,
                                           int             incx,
                                           hipblasStride   stridex,
                                           float*          y,
                                           int             incy,
                                           hipblasStride   stridey,
                                           const float*    param,
                                           hipblasStride   strideParam,
                                           int             batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDrotmStridedBatched(hipblasHandle_t handle,
                                           int             n,
                                           double*         x,
                                           int             incx,
                                           hipblasStride   stridex,
                                           double*         y,
                                           int             incy,
                                           hipblasStride   stridey,
                                           const double*   param,
                                           hipblasStride   strideParam,
                                           int             batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

// Level-1 : rotmg(supported datatypes : float and double )
hipblasStatus_t hipblasSrotmg(
    hipblasHandle_t handle, float* d1, float* d2, float* x1, const float* y1, float* param)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDrotmg(
    hipblasHandle_t handle, double* d1, double* d2, double* x1, const double* y1, double* param)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

// rotmg_batched
hipblasStatus_t hipblasSrotmgBatched(hipblasHandle_t    handle,
                                     float* const       d1[],
                                     float* const       d2[],
                                     float* const       x1[],
                                     const float* const y1[],
                                     float* const       param[],
                                     int                batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDrotmgBatched(hipblasHandle_t     handle,
                                     double* const       d1[],
                                     double* const       d2[],
                                     double* const       x1[],
                                     const double* const y1[],
                                     double* const       param[],
                                     int                 batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

// rotmg_strided_batched
hipblasStatus_t hipblasSrotmgStridedBatched(hipblasHandle_t handle,
                                            float*          d1,
                                            hipblasStride   stride_d1,
                                            float*          d2,
                                            hipblasStride   stride_d2,
                                            float*          x1,
                                            hipblasStride   stride_x1,
                                            const float*    y1,
                                            hipblasStride   stride_y1,
                                            float*          param,
                                            hipblasStride   strideParam,
                                            int             batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDrotmgStridedBatched(hipblasHandle_t handle,
                                            double*         d1,
                                            hipblasStride   stride_d1,
                                            double*         d2,
                                            hipblasStride   stride_d2,
                                            double*         x1,
                                            hipblasStride   stride_x1,
                                            const double*   y1,
                                            hipblasStride   stride_y1,
                                            double*         param,
                                            hipblasStride   strideParam,
                                            int             batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

// Level-1 : scal (supported datatypes : float, double, complex float, complex double)
hipblasStatus_t
    hipblasSscal(hipblasHandle_t handle, int n, const float *alpha, float *x, int incx)
try
{
    // error checks
    if (handle == nullptr || x == nullptr || alpha == nullptr ||
        incx <= 0 || n <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    bool is_dev_ptr = (queryCurrentPtrMode(handle) == HIPBLAS_POINTER_MODE_DEVICE);
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    // As per spec alpha can be device/host memory. In case of device memory *alpha will crash
    float host_alpha = 1.0;
    if (is_dev_ptr) {
        auto hipStatus = hipMemcpy(&host_alpha, alpha, sizeof(float), hipMemcpyDefault);
    } else {
        host_alpha = *alpha;
    }

    onemklSscal(sycl_queue, n, host_alpha, x, incx);
    syclblas_queue_wait(sycl_queue);

    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t
    hipblasDscal(hipblasHandle_t handle, int n, const double *alpha, double *x, int incx)
try
{
    // error checks
    if (handle == nullptr || x == nullptr || alpha == nullptr ||
        incx <= 0 || n <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    bool is_dev_ptr = (queryCurrentPtrMode(handle) == HIPBLAS_POINTER_MODE_DEVICE);
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    // As per spec alpha can be device/host memory. In case of device memory *alpha will crash
    double host_alpha = 1.0;
    if (is_dev_ptr) {
        auto hipStatus = hipMemcpy(&host_alpha, alpha, sizeof(double), hipMemcpyDefault);
    } else {
        host_alpha = *alpha;
    }

    onemklDscal(sycl_queue, n, host_alpha, x, incx);
    syclblas_queue_wait(sycl_queue);

    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t
    hipblasCscal(hipblasHandle_t handle, int n, const hipblasComplex *alpha, hipblasComplex *x, int incx)
try
{
    // error checks
    if (handle == nullptr || x == nullptr || alpha == nullptr ||
        incx <= 0 || n <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    bool is_dev_ptr = (queryCurrentPtrMode(handle) == HIPBLAS_POINTER_MODE_DEVICE);
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    // As per spec alpha can be device/host memory. In case of device memory *alpha will crash
    float _Complex host_alpha = 1.0;
    if (is_dev_ptr) {
        auto hipStatus = hipMemcpy(&host_alpha, alpha, sizeof(float _Complex), hipMemcpyDefault);
    } else {
        host_alpha = *((float _Complex*)alpha);
    }

    onemklCscal(sycl_queue, n, host_alpha, (float _Complex*)x, incx);
    syclblas_queue_wait(sycl_queue);

    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t
    hipblasCsscal(hipblasHandle_t handle, int n, const float *alpha, hipblasComplex *x, int incx)
try
{
    // error checks
    if (handle == nullptr || x == nullptr || alpha == nullptr ||
        incx <= 0 || n <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    bool is_dev_ptr = (queryCurrentPtrMode(handle) == HIPBLAS_POINTER_MODE_DEVICE);
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    // As per spec alpha can be device/host memory. In case of device memory *alpha will crash
    float host_alpha = 1.0;
    if (is_dev_ptr) {
        auto hipStatus = hipMemcpy(&host_alpha, alpha, sizeof(float ), hipMemcpyDefault);
    } else {
        host_alpha = *alpha;
    }
    onemklCsscal(sycl_queue, n, host_alpha, (float _Complex*)x, incx);
    syclblas_queue_wait(sycl_queue);

    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t
    hipblasZscal(hipblasHandle_t handle, int n, const hipblasDoubleComplex *alpha, hipblasDoubleComplex *x, int incx)
try
{
    // error checks
    if (handle == nullptr || x == nullptr || alpha == nullptr ||
        incx <= 0 || n <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    bool is_dev_ptr = (queryCurrentPtrMode(handle) == HIPBLAS_POINTER_MODE_DEVICE);
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    // As per spec alpha can be device/host memory. In case of device memory *alpha will crash
    double _Complex host_alpha = 1.0;
    if (is_dev_ptr) {
        auto hipStatus = hipMemcpy(&host_alpha, alpha, sizeof(double _Complex), hipMemcpyDefault);
    } else {
        host_alpha = *((double _Complex*)alpha);
    }
    onemklZscal(sycl_queue, n, host_alpha, (double _Complex*)x, incx);
    syclblas_queue_wait(sycl_queue);

    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t
    hipblasZdscal(hipblasHandle_t handle, int n, const double *alpha, hipblasDoubleComplex *x, int incx)
try
{
    // error checks
    if (handle == nullptr || x == nullptr || alpha == nullptr ||
        incx <= 0 || n <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    bool is_dev_ptr = (queryCurrentPtrMode(handle) == HIPBLAS_POINTER_MODE_DEVICE);
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    // As per spec alpha can be device/host memory. In case of device memory *alpha will crash
    double host_alpha = 1.0;
    if (is_dev_ptr) {
        auto hipStatus = hipMemcpy(&host_alpha, alpha, sizeof(double), hipMemcpyDefault);
    } else {
        host_alpha = *alpha;
    }
    onemklZdscal(sycl_queue, n, host_alpha, (double _Complex*)x, incx);
    syclblas_queue_wait(sycl_queue);

    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}
// scal_batched
hipblasStatus_t hipblasSscalBatched(
    hipblasHandle_t handle, int n, const float* alpha, float* const x[], int incx, int batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDscalBatched(
    hipblasHandle_t handle, int n, const double* alpha, double* const x[], int incx, int batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCscalBatched(hipblasHandle_t       handle,
                                    int                   n,
                                    const hipblasComplex* alpha,
                                    hipblasComplex* const x[],
                                    int                   incx,
                                    int                   batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZscalBatched(hipblasHandle_t             handle,
                                    int                         n,
                                    const hipblasDoubleComplex* alpha,
                                    hipblasDoubleComplex* const x[],
                                    int                         incx,
                                    int                         batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCsscalBatched(hipblasHandle_t       handle,
                                     int                   n,
                                     const float*          alpha,
                                     hipblasComplex* const x[],
                                     int                   incx,
                                     int                   batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZdscalBatched(hipblasHandle_t             handle,
                                     int                         n,
                                     const double*               alpha,
                                     hipblasDoubleComplex* const x[],
                                     int                         incx,
                                     int                         batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

// scal_strided_batched
hipblasStatus_t hipblasSscalStridedBatched(hipblasHandle_t handle,
                                           int             n,
                                           const float*    alpha,
                                           float*          x,
                                           int             incx,
                                           hipblasStride   stridex,
                                           int             batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDscalStridedBatched(hipblasHandle_t handle,
                                           int             n,
                                           const double*   alpha,
                                           double*         x,
                                           int             incx,
                                           hipblasStride   stridex,
                                           int             batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCscalStridedBatched(hipblasHandle_t       handle,
                                           int                   n,
                                           const hipblasComplex* alpha,
                                           hipblasComplex*       x,
                                           int                   incx,
                                           hipblasStride         stridex,
                                           int                   batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZscalStridedBatched(hipblasHandle_t             handle,
                                           int                         n,
                                           const hipblasDoubleComplex* alpha,
                                           hipblasDoubleComplex*       x,
                                           int                         incx,
                                           hipblasStride               stridex,
                                           int                         batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCsscalStridedBatched(hipblasHandle_t handle,
                                            int             n,
                                            const float*    alpha,
                                            hipblasComplex* x,
                                            int             incx,
                                            hipblasStride   stridex,
                                            int             batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZdscalStridedBatched(hipblasHandle_t       handle,
                                            int                   n,
                                            const double*         alpha,
                                            hipblasDoubleComplex* x,
                                            int                   incx,
                                            hipblasStride         stridex,
                                            int                   batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

// Level-1 : swap (supported datatypes : float, double, complex float, complex double)
hipblasStatus_t hipblasSswap(hipblasHandle_t handle, int n, float* x, int incx, float* y, int incy)
try
{
    // error checks
    if (handle == nullptr || x == nullptr || y == nullptr ||
        incx <= 0 || incy <= 0 || n <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    onemklSswap(sycl_queue, n, x, incx, y, incy);
    syclblas_queue_wait(sycl_queue);
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDswap(hipblasHandle_t handle, int n, double* x, int incx, double* y, int incy)
try
{
    // error checks
    if (handle == nullptr || x == nullptr || y == nullptr ||
        incx <= 0 || incy <= 0 || n <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    onemklDswap(sycl_queue, n, x, incx, y, incy);
    syclblas_queue_wait(sycl_queue);
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCswap(hipblasHandle_t handle, int n, hipblasComplex* x, int incx,
                             hipblasComplex* y, int incy)
try
{
    // error checks
    if (handle == nullptr || x == nullptr || y == nullptr ||
        incx <= 0 || incy <= 0 || n <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    onemklCswap(sycl_queue, n, (float _Complex*)x, incx, (float _Complex*)y, incy);
    syclblas_queue_wait(sycl_queue);
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZswap(hipblasHandle_t handle, int n, hipblasDoubleComplex* x, int incx,
                             hipblasDoubleComplex* y, int incy)
try
{
    // error checks
    if (handle == nullptr || x == nullptr || y == nullptr ||
        incx <= 0 || incy <= 0 || n <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    onemklZswap(sycl_queue, n, (double _Complex*)x, incx, (double _Complex*)y, incy);
    syclblas_queue_wait(sycl_queue);
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}
// swap_batched
hipblasStatus_t hipblasSswapBatched(
    hipblasHandle_t handle, int n, float* x[], int incx, float* y[], int incy, int batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDswapBatched(
    hipblasHandle_t handle, int n, double* x[], int incx, double* y[], int incy, int batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCswapBatched(hipblasHandle_t handle,
                                    int             n,
                                    hipblasComplex* x[],
                                    int             incx,
                                    hipblasComplex* y[],
                                    int             incy,
                                    int             batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZswapBatched(hipblasHandle_t       handle,
                                    int                   n,
                                    hipblasDoubleComplex* x[],
                                    int                   incx,
                                    hipblasDoubleComplex* y[],
                                    int                   incy,
                                    int                   batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

// swap_strided_batched
hipblasStatus_t hipblasSswapStridedBatched(hipblasHandle_t handle,
                                           int             n,
                                           float*          x,
                                           int             incx,
                                           hipblasStride   stridex,
                                           float*          y,
                                           int             incy,
                                           hipblasStride   stridey,
                                           int             batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDswapStridedBatched(hipblasHandle_t handle,
                                           int             n,
                                           double*         x,
                                           int             incx,
                                           hipblasStride   stridex,
                                           double*         y,
                                           int             incy,
                                           hipblasStride   stridey,
                                           int             batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCswapStridedBatched(hipblasHandle_t handle,
                                           int             n,
                                           hipblasComplex* x,
                                           int             incx,
                                           hipblasStride   stridex,
                                           hipblasComplex* y,
                                           int             incy,
                                           hipblasStride   stridey,
                                           int             batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZswapStridedBatched(hipblasHandle_t       handle,
                                           int                   n,
                                           hipblasDoubleComplex* x,
                                           int                   incx,
                                           hipblasStride         stridex,
                                           hipblasDoubleComplex* y,
                                           int                   incy,
                                           hipblasStride         stridey,
                                           int                   batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

/*
 * ===========================================================================
 *    level 2 BLAS
 * ===========================================================================
 */

// Level-2 : gbmv(supported datatypes : float, double, float complex and doule complex)
hipblasStatus_t hipblasSgbmv(hipblasHandle_t handle, hipblasOperation_t trans,
                              int m, int n, int kl, int ku, const float* alpha,
                              const float* AP, int lda, const float* x, int incx,
                              const float* beta, float* y, int incy)
try{
    if (handle == nullptr || alpha == nullptr || AP == nullptr || x == nullptr || beta == nullptr || y == nullptr ||
        m <= 0 || n <= 0 || lda <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    auto is_dev_ptr = (queryCurrentPtrMode(handle) == HIPBLAS_POINTER_MODE_DEVICE);

    // Need to check as alpha and beta can be host/device pointer
    float h_alpha, h_beta;
    if (is_dev_ptr) {
        hipMemcpy(&h_alpha, alpha, sizeof(float), hipMemcpyDefault);
        hipMemcpy(&h_beta, beta, sizeof(float), hipMemcpyDefault);
    } else {
        h_alpha = *((float*)alpha);
        h_beta = *((float*)beta);
    }

    onemklSgbmv(sycl_queue, convert(trans), m, n, kl, ku, h_alpha, AP, lda, x, incx, h_beta, y, incy);

    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDgbmv(hipblasHandle_t handle, hipblasOperation_t trans,
                              int m, int n, int kl, int ku, const double* alpha,
                              const double* AP, int lda, const double* x, int incx,
                              const double* beta, double* y, int incy)
try{
    if (handle == nullptr || alpha == nullptr || AP == nullptr || x == nullptr || beta == nullptr || y == nullptr ||
        m <= 0 || n <= 0 || lda <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    auto is_dev_ptr = (queryCurrentPtrMode(handle) == HIPBLAS_POINTER_MODE_DEVICE);

    double h_alpha, h_beta;
    if (is_dev_ptr) {
        hipMemcpy(&h_alpha, alpha, sizeof(double), hipMemcpyDefault);
        hipMemcpy(&h_beta, beta, sizeof(double), hipMemcpyDefault);
    } else {
        h_alpha = *((double*)alpha);
        h_beta = *((double*)beta);
    }

    onemklDgbmv(sycl_queue, convert(trans), m, n, kl, ku, h_alpha, AP, lda, x, incx, h_beta, y, incy);

    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCgbmv(hipblasHandle_t handle, hipblasOperation_t trans,
                              int m, int n, int kl, int ku, const hipblasComplex* alpha,
                              const hipblasComplex* AP, int lda, const hipblasComplex* x, int incx,
                              const hipblasComplex* beta, hipblasComplex* y, int incy)
try{
    if (handle == nullptr || alpha == nullptr || AP == nullptr || x == nullptr || beta == nullptr || y == nullptr ||
        m <= 0 || n <= 0 || lda <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    auto is_dev_ptr = (queryCurrentPtrMode(handle) == HIPBLAS_POINTER_MODE_DEVICE);

    float _Complex h_alpha, h_beta;
    if (is_dev_ptr) {
        hipMemcpy(&h_alpha, alpha, sizeof(float _Complex), hipMemcpyDefault);
        hipMemcpy(&h_beta, beta, sizeof(float _Complex), hipMemcpyDefault);
    } else {
        h_alpha = *((float _Complex*)alpha);
        h_beta = *((float _Complex*)beta);
    }

    onemklCgbmv(sycl_queue, convert(trans), m, n, kl, ku, h_alpha,
                (const float _Complex *)AP, lda, (const float _Complex *)x, incx,
                 h_beta, (float _Complex *)y, incy);

    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZgbmv(hipblasHandle_t handle, hipblasOperation_t trans,
                              int m, int n, int kl, int ku, const hipblasDoubleComplex* alpha,
                              const hipblasDoubleComplex* AP, int lda, const hipblasDoubleComplex* x, int incx,
                              const hipblasDoubleComplex* beta, hipblasDoubleComplex* y, int incy)
try{
    if (handle == nullptr || alpha == nullptr || AP == nullptr || x == nullptr || beta == nullptr || y == nullptr ||
        m <= 0 || n <= 0 || lda <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    auto is_dev_ptr = (queryCurrentPtrMode(handle) == HIPBLAS_POINTER_MODE_DEVICE);

    double _Complex h_alpha, h_beta;
    if (is_dev_ptr) {
        hipMemcpy(&h_alpha, alpha, sizeof(double _Complex), hipMemcpyDefault);
        hipMemcpy(&h_beta, beta, sizeof(double _Complex), hipMemcpyDefault);
    } else {
        h_alpha = *((double _Complex*)alpha);
        h_beta = *((double _Complex*)beta);
    }

    onemklZgbmv(sycl_queue, convert(trans), m, n, kl, ku, h_alpha,
                (const double _Complex *)AP, lda, (const double _Complex *)x, incx,
                 h_beta, (double _Complex *)y, incy);

    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}
// gbmv_batched
hipblasStatus_t hipblasSgbmvBatched(hipblasHandle_t    handle,
                                    hipblasOperation_t trans,
                                    int                m,
                                    int                n,
                                    int                kl,
                                    int                ku,
                                    const float*       alpha,
                                    const float* const A[],
                                    int                lda,
                                    const float* const x[],
                                    int                incx,
                                    const float*       beta,
                                    float* const       y[],
                                    int                incy,
                                    int                batch_count)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDgbmvBatched(hipblasHandle_t     handle,
                                    hipblasOperation_t  trans,
                                    int                 m,
                                    int                 n,
                                    int                 kl,
                                    int                 ku,
                                    const double*       alpha,
                                    const double* const A[],
                                    int                 lda,
                                    const double* const x[],
                                    int                 incx,
                                    const double*       beta,
                                    double* const       y[],
                                    int                 incy,
                                    int                 batch_count)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCgbmvBatched(hipblasHandle_t             handle,
                                    hipblasOperation_t          trans,
                                    int                         m,
                                    int                         n,
                                    int                         kl,
                                    int                         ku,
                                    const hipblasComplex*       alpha,
                                    const hipblasComplex* const A[],
                                    int                         lda,
                                    const hipblasComplex* const x[],
                                    int                         incx,
                                    const hipblasComplex*       beta,
                                    hipblasComplex* const       y[],
                                    int                         incy,
                                    int                         batch_count)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZgbmvBatched(hipblasHandle_t                   handle,
                                    hipblasOperation_t                trans,
                                    int                               m,
                                    int                               n,
                                    int                               kl,
                                    int                               ku,
                                    const hipblasDoubleComplex*       alpha,
                                    const hipblasDoubleComplex* const A[],
                                    int                               lda,
                                    const hipblasDoubleComplex* const x[],
                                    int                               incx,
                                    const hipblasDoubleComplex*       beta,
                                    hipblasDoubleComplex* const       y[],
                                    int                               incy,
                                    int                               batch_count)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

// gbmv_strided_batched
hipblasStatus_t hipblasSgbmvStridedBatched(hipblasHandle_t    handle,
                                           hipblasOperation_t trans,
                                           int                m,
                                           int                n,
                                           int                kl,
                                           int                ku,
                                           const float*       alpha,
                                           const float*       A,
                                           int                lda,
                                           hipblasStride      stride_a,
                                           const float*       x,
                                           int                incx,
                                           hipblasStride      stride_x,
                                           const float*       beta,
                                           float*             y,
                                           int                incy,
                                           hipblasStride      stride_y,
                                           int                batch_count)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDgbmvStridedBatched(hipblasHandle_t    handle,
                                           hipblasOperation_t trans,
                                           int                m,
                                           int                n,
                                           int                kl,
                                           int                ku,
                                           const double*      alpha,
                                           const double*      A,
                                           int                lda,
                                           hipblasStride      stride_a,
                                           const double*      x,
                                           int                incx,
                                           hipblasStride      stride_x,
                                           const double*      beta,
                                           double*            y,
                                           int                incy,
                                           hipblasStride      stride_y,
                                           int                batch_count)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCgbmvStridedBatched(hipblasHandle_t       handle,
                                           hipblasOperation_t    trans,
                                           int                   m,
                                           int                   n,
                                           int                   kl,
                                           int                   ku,
                                           const hipblasComplex* alpha,
                                           const hipblasComplex* A,
                                           int                   lda,
                                           hipblasStride         stride_a,
                                           const hipblasComplex* x,
                                           int                   incx,
                                           hipblasStride         stride_x,
                                           const hipblasComplex* beta,
                                           hipblasComplex*       y,
                                           int                   incy,
                                           hipblasStride         stride_y,
                                           int                   batch_count)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZgbmvStridedBatched(hipblasHandle_t             handle,
                                           hipblasOperation_t          trans,
                                           int                         m,
                                           int                         n,
                                           int                         kl,
                                           int                         ku,
                                           const hipblasDoubleComplex* alpha,
                                           const hipblasDoubleComplex* A,
                                           int                         lda,
                                           hipblasStride               stride_a,
                                           const hipblasDoubleComplex* x,
                                           int                         incx,
                                           hipblasStride               stride_x,
                                           const hipblasDoubleComplex* beta,
                                           hipblasDoubleComplex*       y,
                                           int                         incy,
                                           hipblasStride               stride_y,
                                           int                         batch_count)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

// Level-2 : gemv(supported datatypes : float, double, float complex and doule complex)
hipblasStatus_t hipblasSgemv(hipblasHandle_t handle, hipblasOperation_t trans, int m, int n,
                             const float* alpha, const float* AP, int lda, const float* x, int incx,
                             const float* beta, float* y, int incy)
try
{
    if (handle == nullptr || alpha == nullptr || AP == nullptr || x == nullptr || beta == nullptr || y == nullptr ||
        m <= 0 || n <= 0 || lda <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    auto is_dev_ptr = (queryCurrentPtrMode(handle) == HIPBLAS_POINTER_MODE_DEVICE);

    float h_alpha, h_beta;
    if (is_dev_ptr) {
        hipMemcpy(&h_alpha, alpha, sizeof(float), hipMemcpyDefault);
        hipMemcpy(&h_beta, beta, sizeof(float), hipMemcpyDefault);
    } else {
        h_alpha = *((float*)alpha);
        h_beta = *((float*)beta);
    }

    onemklSgemv(sycl_queue, convert(trans), m, n, h_alpha, AP, lda, x, incx, h_beta, y, incy);

    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDgemv(hipblasHandle_t handle, hipblasOperation_t trans, int m, int n,
                             const double* alpha, const double* AP, int lda, const double* x, int incx,
                             const double* beta, double* y, int incy)
try
{
    if (handle == nullptr || alpha == nullptr || AP == nullptr || x == nullptr || beta == nullptr || y == nullptr ||
        m <= 0 || n <= 0 || lda <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    auto is_dev_ptr = (queryCurrentPtrMode(handle) == HIPBLAS_POINTER_MODE_DEVICE);

    double h_alpha, h_beta;
    if (is_dev_ptr) {
        hipMemcpy(&h_alpha, alpha, sizeof(double), hipMemcpyDefault);
        hipMemcpy(&h_beta, beta, sizeof(double), hipMemcpyDefault);
    } else {
        h_alpha = *((double*)alpha);
        h_beta = *((double*)beta);
    }

    onemklDgemv(sycl_queue, convert(trans), m, n, h_alpha, AP, lda, x, incx, h_beta, y, incy);

    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCgemv(hipblasHandle_t handle, hipblasOperation_t trans,
                              int m, int n, const hipblasComplex* alpha,
                              const hipblasComplex* AP, int lda, const hipblasComplex* x, int incx,
                              const hipblasComplex* beta, hipblasComplex* y, int incy)
try{
    if (handle == nullptr || alpha == nullptr || AP == nullptr || x == nullptr || beta == nullptr || y == nullptr ||
        m <= 0 || n <= 0 || lda <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    auto is_dev_ptr = (queryCurrentPtrMode(handle) == HIPBLAS_POINTER_MODE_DEVICE);

    float _Complex h_alpha, h_beta;
    if (is_dev_ptr) {
        hipMemcpy(&h_alpha, alpha, sizeof(float _Complex), hipMemcpyDefault);
        hipMemcpy(&h_beta, beta, sizeof(float _Complex), hipMemcpyDefault);
    } else {
        h_alpha = *((float _Complex*)alpha);
        h_beta = *((float _Complex*)beta);
    }

    onemklCgemv(sycl_queue, convert(trans), m, n, h_alpha,
                (const float _Complex *)AP, lda, (const float _Complex *)x, incx,
                 h_beta, (float _Complex *)y, incy);

    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZgemv(hipblasHandle_t handle, hipblasOperation_t trans,
                              int m, int n, const hipblasDoubleComplex* alpha,
                              const hipblasDoubleComplex* AP, int lda, const hipblasDoubleComplex* x, int incx,
                              const hipblasDoubleComplex* beta, hipblasDoubleComplex* y, int incy)
try{
    if (handle == nullptr || alpha == nullptr || AP == nullptr || x == nullptr || beta == nullptr || y == nullptr ||
        m <= 0 || n <= 0 || lda <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    auto is_dev_ptr = (queryCurrentPtrMode(handle) == HIPBLAS_POINTER_MODE_DEVICE);

    double _Complex h_alpha, h_beta;
    if (is_dev_ptr) {
        hipMemcpy(&h_alpha, alpha, sizeof(double _Complex), hipMemcpyDefault);
        hipMemcpy(&h_beta, beta, sizeof(double _Complex), hipMemcpyDefault);
    } else {
        h_alpha = *((double _Complex*)alpha);
        h_beta = *((double _Complex*)beta);
    }

    onemklZgemv(sycl_queue, convert(trans), m, n, h_alpha,
                (const double _Complex *)AP, lda, (const double _Complex *)x, incx,
                 h_beta, (double _Complex *)y, incy);

    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}
// gemv_batched
hipblasStatus_t hipblasSgemvBatched(hipblasHandle_t    handle,
                                    hipblasOperation_t trans,
                                    int                m,
                                    int                n,
                                    const float*       alpha,
                                    const float* const A[],
                                    int                lda,
                                    const float* const x[],
                                    int                incx,
                                    const float*       beta,
                                    float* const       y[],
                                    int                incy,
                                    int                batchCount)
try
{
return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDgemvBatched(hipblasHandle_t     handle,
                                    hipblasOperation_t  trans,
                                    int                 m,
                                    int                 n,
                                    const double*       alpha,
                                    const double* const A[],
                                    int                 lda,
                                    const double* const x[],
                                    int                 incx,
                                    const double*       beta,
                                    double* const       y[],
                                    int                 incy,
                                    int                 batchCount)
try
{
return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCgemvBatched(hipblasHandle_t             handle,
                                    hipblasOperation_t          trans,
                                    int                         m,
                                    int                         n,
                                    const hipblasComplex*       alpha,
                                    const hipblasComplex* const A[],
                                    int                         lda,
                                    const hipblasComplex* const x[],
                                    int                         incx,
                                    const hipblasComplex*       beta,
                                    hipblasComplex* const       y[],
                                    int                         incy,
                                    int                         batchCount)
try
{
return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZgemvBatched(hipblasHandle_t                   handle,
                                    hipblasOperation_t                trans,
                                    int                               m,
                                    int                               n,
                                    const hipblasDoubleComplex*       alpha,
                                    const hipblasDoubleComplex* const A[],
                                    int                               lda,
                                    const hipblasDoubleComplex* const x[],
                                    int                               incx,
                                    const hipblasDoubleComplex*       beta,
                                    hipblasDoubleComplex* const       y[],
                                    int                               incy,
                                    int                               batchCount)
try
{
return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

// gemv_strided_batched
hipblasStatus_t hipblasSgemvStridedBatched(hipblasHandle_t    handle,
                                           hipblasOperation_t trans,
                                           int                m,
                                           int                n,
                                           const float*       alpha,
                                           const float*       A,
                                           int                lda,
                                           hipblasStride      strideA,
                                           const float*       x,
                                           int                incx,
                                           hipblasStride      stridex,
                                           const float*       beta,
                                           float*             y,
                                           int                incy,
                                           hipblasStride      stridey,
                                           int                batchCount)
try
{
return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDgemvStridedBatched(hipblasHandle_t    handle,
                                           hipblasOperation_t trans,
                                           int                m,
                                           int                n,
                                           const double*      alpha,
                                           const double*      A,
                                           int                lda,
                                           hipblasStride      strideA,
                                           const double*      x,
                                           int                incx,
                                           hipblasStride      stridex,
                                           const double*      beta,
                                           double*            y,
                                           int                incy,
                                           hipblasStride      stridey,
                                           int                batchCount)
try
{
return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCgemvStridedBatched(hipblasHandle_t       handle,
                                           hipblasOperation_t    trans,
                                           int                   m,
                                           int                   n,
                                           const hipblasComplex* alpha,
                                           const hipblasComplex* A,
                                           int                   lda,
                                           hipblasStride         strideA,
                                           const hipblasComplex* x,
                                           int                   incx,
                                           hipblasStride         stridex,
                                           const hipblasComplex* beta,
                                           hipblasComplex*       y,
                                           int                   incy,
                                           hipblasStride         stridey,
                                           int                   batchCount)
try
{
return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZgemvStridedBatched(hipblasHandle_t             handle,
                                           hipblasOperation_t          trans,
                                           int                         m,
                                           int                         n,
                                           const hipblasDoubleComplex* alpha,
                                           const hipblasDoubleComplex* A,
                                           int                         lda,
                                           hipblasStride               strideA,
                                           const hipblasDoubleComplex* x,
                                           int                         incx,
                                           hipblasStride               stridex,
                                           const hipblasDoubleComplex* beta,
                                           hipblasDoubleComplex*       y,
                                           int                         incy,
                                           hipblasStride               stridey,
                                           int                         batchCount)
try
{
return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

// Level-2 : ger(supported datatypes : float, double, float complex and doule complex)
hipblasStatus_t hipblasSger(hipblasHandle_t handle, int m, int n, const float* alpha,
                            const float* x, int incx, const float* y, int incy,
                            float* AP, int lda)
try
{
    if (handle == nullptr || alpha == nullptr || x == nullptr || y == nullptr || AP == nullptr ||
        m <= 0 || n <= 0 || lda <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }    
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    auto is_dev_ptr = (queryCurrentPtrMode(handle) == HIPBLAS_POINTER_MODE_DEVICE);

    float h_alpha;
    if (is_dev_ptr) {
        hipMemcpy(&h_alpha, alpha, sizeof(float), hipMemcpyDefault);
    } else {
        h_alpha = *((float*)alpha);
    }
    onemklSger(sycl_queue, m, n, h_alpha, x, incx, y, incy, AP, lda);
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDger(hipblasHandle_t handle, int m, int n, const double* alpha,
                            const double* x, int incx, const double* y, int incy,
                            double* AP, int lda)
try
{
    if (handle == nullptr || alpha == nullptr || x == nullptr || y == nullptr || AP == nullptr ||
        m <= 0 || n <= 0 || lda <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }    
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    auto is_dev_ptr = (queryCurrentPtrMode(handle) == HIPBLAS_POINTER_MODE_DEVICE);

    double h_alpha;
    if (is_dev_ptr) {
        hipMemcpy(&h_alpha, alpha, sizeof(double), hipMemcpyDefault);
    } else {
        h_alpha = *((double*)alpha);
    }
    onemklDger(sycl_queue, m, n, h_alpha, x, incx, y, incy, AP, lda);
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCgerc(hipblasHandle_t handle, int m, int n, const hipblasComplex* alpha,
                            const hipblasComplex* x, int incx, const hipblasComplex* y, int incy,
                            hipblasComplex* AP, int lda)
try
{
    if (handle == nullptr || alpha == nullptr || x == nullptr || y == nullptr || AP == nullptr ||
        m <= 0 || n <= 0 || lda <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }    
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    auto is_dev_ptr = (queryCurrentPtrMode(handle) == HIPBLAS_POINTER_MODE_DEVICE);

    float _Complex h_alpha;
    if (is_dev_ptr) {
        hipMemcpy(&h_alpha, alpha, sizeof(float _Complex), hipMemcpyDefault);
    } else {
        h_alpha = *((float _Complex*)alpha);
    }
    onemklCgerc(sycl_queue, m, n, h_alpha, (const float _Complex*)x, incx, (const float _Complex*)y, incy,
                (float _Complex*)AP, lda);
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCgeru(hipblasHandle_t handle, int m, int n, const hipblasComplex* alpha,
                            const hipblasComplex* x, int incx, const hipblasComplex* y, int incy,
                            hipblasComplex* AP, int lda)
try
{
    if (handle == nullptr || alpha == nullptr || x == nullptr || y == nullptr || AP == nullptr ||
        m <= 0 || n <= 0 || lda <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }    
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    auto is_dev_ptr = (queryCurrentPtrMode(handle) == HIPBLAS_POINTER_MODE_DEVICE);

    float _Complex h_alpha;
    if (is_dev_ptr) {
        hipMemcpy(&h_alpha, alpha, sizeof(float _Complex), hipMemcpyDefault);
    } else {
        h_alpha = *((float _Complex*)alpha);
    }
    onemklCgeru(sycl_queue, m, n, h_alpha, (const float _Complex*)x, incx, (const float _Complex*)y, incy,
                (float _Complex*)AP, lda);
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZgerc(hipblasHandle_t handle, int m, int n, const hipblasDoubleComplex* alpha,
                            const hipblasDoubleComplex* x, int incx, const hipblasDoubleComplex* y, int incy,
                            hipblasDoubleComplex* AP, int lda)
try
{
    if (handle == nullptr || alpha == nullptr || x == nullptr || y == nullptr || AP == nullptr ||
        m <= 0 || n <= 0 || lda <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }    
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    auto is_dev_ptr = (queryCurrentPtrMode(handle) == HIPBLAS_POINTER_MODE_DEVICE);

    double _Complex h_alpha;
    if (is_dev_ptr) {
        hipMemcpy(&h_alpha, alpha, sizeof(double _Complex), hipMemcpyDefault);
    } else {
        h_alpha = *((double _Complex*)alpha);
    }
    onemklZgerc(sycl_queue, m, n, h_alpha, (const double _Complex*)x, incx, (const double _Complex*)y, incy,
                (double _Complex*)AP, lda);
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZgeru(hipblasHandle_t handle, int m, int n, const hipblasDoubleComplex* alpha,
                            const hipblasDoubleComplex* x, int incx, const hipblasDoubleComplex* y, int incy,
                            hipblasDoubleComplex* AP, int lda)
try
{
    if (handle == nullptr || alpha == nullptr || x == nullptr || y == nullptr || AP == nullptr ||
        m <= 0 || n <= 0 || lda <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }    
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    auto is_dev_ptr = (queryCurrentPtrMode(handle) == HIPBLAS_POINTER_MODE_DEVICE);

    double _Complex h_alpha;
    if (is_dev_ptr) {
        hipMemcpy(&h_alpha, alpha, sizeof(double _Complex), hipMemcpyDefault);
    } else {
        h_alpha = *((double _Complex*)alpha);
    }
    onemklZgeru(sycl_queue, m, n, h_alpha, (const double _Complex*)x, incx, (const double _Complex*)y, incy,
                (double _Complex*)AP, lda);
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}
// ger_batched
hipblasStatus_t hipblasSgerBatched(hipblasHandle_t    handle,
                                   int                m,
                                   int                n,
                                   const float*       alpha,
                                   const float* const x[],
                                   int                incx,
                                   const float* const y[],
                                   int                incy,
                                   float* const       A[],
                                   int                lda,
                                   int                batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDgerBatched(hipblasHandle_t     handle,
                                   int                 m,
                                   int                 n,
                                   const double*       alpha,
                                   const double* const x[],
                                   int                 incx,
                                   const double* const y[],
                                   int                 incy,
                                   double* const       A[],
                                   int                 lda,
                                   int                 batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCgeruBatched(hipblasHandle_t             handle,
                                    int                         m,
                                    int                         n,
                                    const hipblasComplex*       alpha,
                                    const hipblasComplex* const x[],
                                    int                         incx,
                                    const hipblasComplex* const y[],
                                    int                         incy,
                                    hipblasComplex* const       A[],
                                    int                         lda,
                                    int                         batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCgercBatched(hipblasHandle_t             handle,
                                    int                         m,
                                    int                         n,
                                    const hipblasComplex*       alpha,
                                    const hipblasComplex* const x[],
                                    int                         incx,
                                    const hipblasComplex* const y[],
                                    int                         incy,
                                    hipblasComplex* const       A[],
                                    int                         lda,
                                    int                         batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZgeruBatched(hipblasHandle_t                   handle,
                                    int                               m,
                                    int                               n,
                                    const hipblasDoubleComplex*       alpha,
                                    const hipblasDoubleComplex* const x[],
                                    int                               incx,
                                    const hipblasDoubleComplex* const y[],
                                    int                               incy,
                                    hipblasDoubleComplex* const       A[],
                                    int                               lda,
                                    int                               batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZgercBatched(hipblasHandle_t                   handle,
                                    int                               m,
                                    int                               n,
                                    const hipblasDoubleComplex*       alpha,
                                    const hipblasDoubleComplex* const x[],
                                    int                               incx,
                                    const hipblasDoubleComplex* const y[],
                                    int                               incy,
                                    hipblasDoubleComplex* const       A[],
                                    int                               lda,
                                    int                               batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

// ger_strided_batched
hipblasStatus_t hipblasSgerStridedBatched(hipblasHandle_t handle,
                                          int             m,
                                          int             n,
                                          const float*    alpha,
                                          const float*    x,
                                          int             incx,
                                          hipblasStride   stridex,
                                          const float*    y,
                                          int             incy,
                                          hipblasStride   stridey,
                                          float*          A,
                                          int             lda,
                                          hipblasStride   strideA,
                                          int             batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDgerStridedBatched(hipblasHandle_t handle,
                                          int             m,
                                          int             n,
                                          const double*   alpha,
                                          const double*   x,
                                          int             incx,
                                          hipblasStride   stridex,
                                          const double*   y,
                                          int             incy,
                                          hipblasStride   stridey,
                                          double*         A,
                                          int             lda,
                                          hipblasStride   strideA,
                                          int             batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCgeruStridedBatched(hipblasHandle_t       handle,
                                           int                   m,
                                           int                   n,
                                           const hipblasComplex* alpha,
                                           const hipblasComplex* x,
                                           int                   incx,
                                           hipblasStride         stridex,
                                           const hipblasComplex* y,
                                           int                   incy,
                                           hipblasStride         stridey,
                                           hipblasComplex*       A,
                                           int                   lda,
                                           hipblasStride         strideA,
                                           int                   batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCgercStridedBatched(hipblasHandle_t       handle,
                                           int                   m,
                                           int                   n,
                                           const hipblasComplex* alpha,
                                           const hipblasComplex* x,
                                           int                   incx,
                                           hipblasStride         stridex,
                                           const hipblasComplex* y,
                                           int                   incy,
                                           hipblasStride         stridey,
                                           hipblasComplex*       A,
                                           int                   lda,
                                           hipblasStride         strideA,
                                           int                   batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZgeruStridedBatched(hipblasHandle_t             handle,
                                           int                         m,
                                           int                         n,
                                           const hipblasDoubleComplex* alpha,
                                           const hipblasDoubleComplex* x,
                                           int                         incx,
                                           hipblasStride               stridex,
                                           const hipblasDoubleComplex* y,
                                           int                         incy,
                                           hipblasStride               stridey,
                                           hipblasDoubleComplex*       A,
                                           int                         lda,
                                           hipblasStride               strideA,
                                           int                         batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZgercStridedBatched(hipblasHandle_t             handle,
                                           int                         m,
                                           int                         n,
                                           const hipblasDoubleComplex* alpha,
                                           const hipblasDoubleComplex* x,
                                           int                         incx,
                                           hipblasStride               stridex,
                                           const hipblasDoubleComplex* y,
                                           int                         incy,
                                           hipblasStride               stridey,
                                           hipblasDoubleComplex*       A,
                                           int                         lda,
                                           hipblasStride               strideA,
                                           int                         batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

// Level-2 : hbmv(supported datatypes : float complex and doule complex)
hipblasStatus_t hipblasChbmv(hipblasHandle_t handle, hipblasFillMode_t uplo,
                            int n, int k, const hipblasComplex* alpha,
                            const hipblasComplex* AP, int lda,
                            const hipblasComplex* x, int incx,
                             const hipblasComplex* beta, hipblasComplex* y, int incy)
try
{
    if (handle == nullptr || alpha == nullptr || AP == nullptr || x == nullptr || beta == nullptr || y == nullptr ||
        k <= 0 || n <= 0 || lda <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    auto is_dev_ptr = (queryCurrentPtrMode(handle) == HIPBLAS_POINTER_MODE_DEVICE);

    float _Complex h_alpha, h_beta;
    if (is_dev_ptr) {
        hipMemcpy(&h_alpha, alpha, sizeof(float _Complex), hipMemcpyDefault);
        hipMemcpy(&h_beta, beta, sizeof(float _Complex), hipMemcpyDefault);
    } else {
        h_alpha = *((float _Complex*)alpha);
        h_beta = *((float _Complex*)beta);
    }

    onemklChbmv(sycl_queue, convert(uplo), n,k, h_alpha, (const float _Complex*)AP, lda, (const float _Complex*)x, incx, 
                h_beta, (float _Complex*)y, incy);
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZhbmv(hipblasHandle_t handle, hipblasFillMode_t uplo,
                            int n, int k, const hipblasDoubleComplex* alpha,
                            const hipblasDoubleComplex* AP, int lda,
                            const hipblasDoubleComplex* x, int incx,
                             const hipblasDoubleComplex* beta, hipblasDoubleComplex* y, int incy)
try
{
    if (handle == nullptr || alpha == nullptr || AP == nullptr || x == nullptr || beta == nullptr || y == nullptr ||
        k <= 0 || n <= 0 || lda <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    auto is_dev_ptr = (queryCurrentPtrMode(handle) == HIPBLAS_POINTER_MODE_DEVICE);

    double _Complex h_alpha, h_beta;
    if (is_dev_ptr) {
        hipMemcpy(&h_alpha, alpha, sizeof(double _Complex), hipMemcpyDefault);
        hipMemcpy(&h_beta, beta, sizeof(double _Complex), hipMemcpyDefault);
    } else {
        h_alpha = *((double _Complex*)alpha);
        h_beta = *((double _Complex*)beta);
    }

    onemklZhbmv(sycl_queue, convert(uplo), n,k, h_alpha, (const double _Complex*)AP, lda, (const double _Complex*)x, incx, 
                h_beta, (double _Complex*)y, incy);
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}
// hbmv_batched
hipblasStatus_t hipblasChbmvBatched(hipblasHandle_t             handle,
                                    hipblasFillMode_t           uplo,
                                    int                         n,
                                    int                         k,
                                    const hipblasComplex*       alpha,
                                    const hipblasComplex* const A[],
                                    int                         lda,
                                    const hipblasComplex* const x[],
                                    int                         incx,
                                    const hipblasComplex*       beta,
                                    hipblasComplex* const       y[],
                                    int                         incy,
                                    int                         batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZhbmvBatched(hipblasHandle_t                   handle,
                                    hipblasFillMode_t                 uplo,
                                    int                               n,
                                    int                               k,
                                    const hipblasDoubleComplex*       alpha,
                                    const hipblasDoubleComplex* const A[],
                                    int                               lda,
                                    const hipblasDoubleComplex* const x[],
                                    int                               incx,
                                    const hipblasDoubleComplex*       beta,
                                    hipblasDoubleComplex* const       y[],
                                    int                               incy,
                                    int                               batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

// hbmv_strided_batched
hipblasStatus_t hipblasChbmvStridedBatched(hipblasHandle_t       handle,
                                           hipblasFillMode_t     uplo,
                                           int                   n,
                                           int                   k,
                                           const hipblasComplex* alpha,
                                           const hipblasComplex* A,
                                           int                   lda,
                                           hipblasStride         strideA,
                                           const hipblasComplex* x,
                                           int                   incx,
                                           hipblasStride         stridex,
                                           const hipblasComplex* beta,
                                           hipblasComplex*       y,
                                           int                   incy,
                                           hipblasStride         stridey,
                                           int                   batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZhbmvStridedBatched(hipblasHandle_t             handle,
                                           hipblasFillMode_t           uplo,
                                           int                         n,
                                           int                         k,
                                           const hipblasDoubleComplex* alpha,
                                           const hipblasDoubleComplex* A,
                                           int                         lda,
                                           hipblasStride               strideA,
                                           const hipblasDoubleComplex* x,
                                           int                         incx,
                                           hipblasStride               stridex,
                                           const hipblasDoubleComplex* beta,
                                           hipblasDoubleComplex*       y,
                                           int                         incy,
                                           hipblasStride               stridey,
                                           int                         batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

// Level-2 : hemv(supported datatypes : float complex and doule complex)
hipblasStatus_t hipblasChemv(hipblasHandle_t handle, hipblasFillMode_t uplo, int n,
                            const hipblasComplex* alpha, const hipblasComplex* AP,
                            int lda, const hipblasComplex* x, int incx,
                            const hipblasComplex* beta, hipblasComplex* y, int incy)
try
{
    if (handle == nullptr || alpha == nullptr || AP == nullptr || x == nullptr || beta == nullptr || y == nullptr ||
        n <= 0 || lda <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    auto is_dev_ptr = (queryCurrentPtrMode(handle) == HIPBLAS_POINTER_MODE_DEVICE);

    float _Complex h_alpha, h_beta;
    if (is_dev_ptr) {
        hipMemcpy(&h_alpha, alpha, sizeof(float _Complex), hipMemcpyDefault);
        hipMemcpy(&h_beta, beta, sizeof(float _Complex), hipMemcpyDefault);
    } else {
        h_alpha = *((float _Complex*)alpha);
        h_beta = *((float _Complex*)beta);
    }

    onemklChemv(sycl_queue, convert(uplo), n, h_alpha, (const float _Complex*)AP, lda,
                (const float _Complex*)x, incx, h_beta, (float _Complex*)y, incy);
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZhemv(hipblasHandle_t handle, hipblasFillMode_t uplo, int n,
                            const hipblasDoubleComplex* alpha, const hipblasDoubleComplex* AP,
                            int lda, const hipblasDoubleComplex* x, int incx,
                            const hipblasDoubleComplex* beta, hipblasDoubleComplex* y, int incy)
try
{
    if (handle == nullptr || alpha == nullptr || AP == nullptr || x == nullptr || beta == nullptr || y == nullptr ||
        n <= 0 || lda <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    auto is_dev_ptr = (queryCurrentPtrMode(handle) == HIPBLAS_POINTER_MODE_DEVICE);

    double _Complex h_alpha, h_beta;
    if (is_dev_ptr) {
        hipMemcpy(&h_alpha, alpha, sizeof(double _Complex), hipMemcpyDefault);
        hipMemcpy(&h_beta, beta, sizeof(double _Complex), hipMemcpyDefault);
    } else {
        h_alpha = *((double _Complex*)alpha);
        h_beta = *((double _Complex*)beta);
    }

    onemklZhemv(sycl_queue, convert(uplo), n, h_alpha, (const double _Complex*)AP, lda,
                (const double _Complex*)x, incx, h_beta, (double _Complex*)y, incy); 
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}
// hemv_batched
hipblasStatus_t hipblasChemvBatched(hipblasHandle_t             handle,
                                    hipblasFillMode_t           uplo,
                                    int                         n,
                                    const hipblasComplex*       alpha,
                                    const hipblasComplex* const A[],
                                    int                         lda,
                                    const hipblasComplex* const x[],
                                    int                         incx,
                                    const hipblasComplex*       beta,
                                    hipblasComplex* const       y[],
                                    int                         incy,
                                    int                         batch_count)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZhemvBatched(hipblasHandle_t                   handle,
                                    hipblasFillMode_t                 uplo,
                                    int                               n,
                                    const hipblasDoubleComplex*       alpha,
                                    const hipblasDoubleComplex* const A[],
                                    int                               lda,
                                    const hipblasDoubleComplex* const x[],
                                    int                               incx,
                                    const hipblasDoubleComplex*       beta,
                                    hipblasDoubleComplex* const       y[],
                                    int                               incy,
                                    int                               batch_count)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

// hemv_strided_batched
hipblasStatus_t hipblasChemvStridedBatched(hipblasHandle_t       handle,
                                           hipblasFillMode_t     uplo,
                                           int                   n,
                                           const hipblasComplex* alpha,
                                           const hipblasComplex* A,
                                           int                   lda,
                                           hipblasStride         stride_a,
                                           const hipblasComplex* x,
                                           int                   incx,
                                           hipblasStride         stride_x,
                                           const hipblasComplex* beta,
                                           hipblasComplex*       y,
                                           int                   incy,
                                           hipblasStride         stride_y,
                                           int                   batch_count)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZhemvStridedBatched(hipblasHandle_t             handle,
                                           hipblasFillMode_t           uplo,
                                           int                         n,
                                           const hipblasDoubleComplex* alpha,
                                           const hipblasDoubleComplex* A,
                                           int                         lda,
                                           hipblasStride               stride_a,
                                           const hipblasDoubleComplex* x,
                                           int                         incx,
                                           hipblasStride               stride_x,
                                           const hipblasDoubleComplex* beta,
                                           hipblasDoubleComplex*       y,
                                           int                         incy,
                                           hipblasStride               stride_y,
                                           int                         batch_count)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

// Level-2 : her(supported datatypes : float complex and doule complex)
hipblasStatus_t hipblasCher(hipblasHandle_t handle, hipblasFillMode_t uplo, int n,
                            const float* alpha, const hipblasComplex* x, int incx,
                            hipblasComplex* AP, int lda)
try
{
    if (handle == nullptr || alpha == nullptr || AP == nullptr || x == nullptr ||
        n <= 0 || lda <= 0 ) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    auto is_dev_ptr = (queryCurrentPtrMode(handle) == HIPBLAS_POINTER_MODE_DEVICE);

    float h_alpha;
    if (is_dev_ptr) {
        hipMemcpy(&h_alpha, alpha, sizeof(float), hipMemcpyDefault);
    } else {
        h_alpha = *((float*)alpha);
    }
    // workaround:
    // {AP := AP + alpha*x*x**H} -> for alpha=0 AP's value should not change
    // but incase of oneMKL it changes to zero. Once bug is fixed workaround should be removed
    if (h_alpha == 0) {
        return HIPBLAS_STATUS_SUCCESS;
    }
    onemklCher(sycl_queue, convert(uplo), n, h_alpha, (const float _Complex*)x, incx, (float _Complex*)AP, lda);
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZher(hipblasHandle_t handle, hipblasFillMode_t uplo, int n,
                            const double* alpha, const hipblasDoubleComplex* x, int incx,
                            hipblasDoubleComplex* AP, int lda)
try
{
    if (handle == nullptr || alpha == nullptr || AP == nullptr || x == nullptr ||
        n <= 0 || lda <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    auto is_dev_ptr = (queryCurrentPtrMode(handle) == HIPBLAS_POINTER_MODE_DEVICE);

    double h_alpha;
    if (is_dev_ptr) {
        hipMemcpy(&h_alpha, alpha, sizeof(double), hipMemcpyDefault);
    } else {
        h_alpha = *((double*)alpha);
    }
    // workaround:
    // {AP := AP + alpha*x*x**H} -> for alpha=0 AP's value should not change
    // but incase of oneMKL it changes to zero. Once bug is fixed workaround should be removed
    if (h_alpha == 0) {
        return HIPBLAS_STATUS_SUCCESS;
    }
    onemklZher(sycl_queue, convert(uplo), n, h_alpha, (const double _Complex*)x, incx, (double _Complex*)AP, lda);
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}
// her_batched
hipblasStatus_t hipblasCherBatched(hipblasHandle_t             handle,
                                   hipblasFillMode_t           uplo,
                                   int                         n,
                                   const float*                alpha,
                                   const hipblasComplex* const x[],
                                   int                         incx,
                                   hipblasComplex* const       A[],
                                   int                         lda,
                                   int                         batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZherBatched(hipblasHandle_t                   handle,
                                   hipblasFillMode_t                 uplo,
                                   int                               n,
                                   const double*                     alpha,
                                   const hipblasDoubleComplex* const x[],
                                   int                               incx,
                                   hipblasDoubleComplex* const       A[],
                                   int                               lda,
                                   int                               batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

// her_strided_batched
hipblasStatus_t hipblasCherStridedBatched(hipblasHandle_t       handle,
                                          hipblasFillMode_t     uplo,
                                          int                   n,
                                          const float*          alpha,
                                          const hipblasComplex* x,
                                          int                   incx,
                                          hipblasStride         stridex,
                                          hipblasComplex*       A,
                                          int                   lda,
                                          hipblasStride         strideA,
                                          int                   batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZherStridedBatched(hipblasHandle_t             handle,
                                          hipblasFillMode_t           uplo,
                                          int                         n,
                                          const double*               alpha,
                                          const hipblasDoubleComplex* x,
                                          int                         incx,
                                          hipblasStride               stridex,
                                          hipblasDoubleComplex*       A,
                                          int                         lda,
                                          hipblasStride               strideA,
                                          int                         batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

// Level-2 : her2(supported datatypes : float complex and doule complex)
hipblasStatus_t hipblasCher2(hipblasHandle_t handle, hipblasFillMode_t uplo, int n,
                            const hipblasComplex* alpha, const hipblasComplex* x, int incx,
                            const hipblasComplex* y, int incy, hipblasComplex* AP, int lda)
try
{
    if (handle == nullptr || alpha == nullptr || AP == nullptr || x == nullptr || y == nullptr ||
        n <= 0 || lda <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    auto is_dev_ptr = (queryCurrentPtrMode(handle) == HIPBLAS_POINTER_MODE_DEVICE);

    float _Complex h_alpha;
    if (is_dev_ptr) {
        hipMemcpy(&h_alpha, alpha, sizeof(float _Complex), hipMemcpyDefault);
    } else {
        h_alpha = *((float _Complex*)alpha);
    }

    // workaround:
    // {AP := AP + alpha*x*y**H + conj(alpha)*y*x**H} -> for alpha=0 AP's value should not change
    // but incase of oneMKL it changes to zero. Once bug is fixed workaround should be removed
    if (h_alpha == 0) {
        return HIPBLAS_STATUS_SUCCESS;
    }

    onemklCher2(sycl_queue, convert(uplo), n, h_alpha, (const float _Complex*)x, incx,
                (const float _Complex*)y, incy, (float _Complex*)AP, lda);
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZher2(hipblasHandle_t handle, hipblasFillMode_t uplo, int n,
                            const hipblasDoubleComplex* alpha, const hipblasDoubleComplex* x, int incx,
                            const hipblasDoubleComplex* y, int incy, hipblasDoubleComplex* AP, int lda)
try
{
    if (handle == nullptr || alpha == nullptr || AP == nullptr || x == nullptr || y == nullptr ||
        n <= 0 || lda <= 0 || incx <= 0 || incy <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    auto is_dev_ptr = (queryCurrentPtrMode(handle) == HIPBLAS_POINTER_MODE_DEVICE);

    double _Complex h_alpha;
    if (is_dev_ptr) {
        hipMemcpy(&h_alpha, alpha, sizeof(double _Complex), hipMemcpyDefault);
    } else {
        h_alpha = *((double _Complex*)alpha);
    }
    // workaround:
    // {AP := AP + alpha*x*y**H + conj(alpha)*y*x**H} -> for alpha=0 AP's value should not change
    // but incase of oneMKL it changes to zero. Once bug is fixed workaround should be removed
    if (h_alpha == 0) {
        return HIPBLAS_STATUS_SUCCESS;
    }
    onemklZher2(sycl_queue, convert(uplo), n, h_alpha, (const double _Complex*)x, incx,
                (const double _Complex*)y, incy, (double _Complex*)AP, lda);
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}
// her2_batched
hipblasStatus_t hipblasCher2Batched(hipblasHandle_t             handle,
                                    hipblasFillMode_t           uplo,
                                    int                         n,
                                    const hipblasComplex*       alpha,
                                    const hipblasComplex* const x[],
                                    int                         incx,
                                    const hipblasComplex* const y[],
                                    int                         incy,
                                    hipblasComplex* const       A[],
                                    int                         lda,
                                    int                         batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZher2Batched(hipblasHandle_t                   handle,
                                    hipblasFillMode_t                 uplo,
                                    int                               n,
                                    const hipblasDoubleComplex*       alpha,
                                    const hipblasDoubleComplex* const x[],
                                    int                               incx,
                                    const hipblasDoubleComplex* const y[],
                                    int                               incy,
                                    hipblasDoubleComplex* const       A[],
                                    int                               lda,
                                    int                               batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

// her2_strided_batched
hipblasStatus_t hipblasCher2StridedBatched(hipblasHandle_t       handle,
                                           hipblasFillMode_t     uplo,
                                           int                   n,
                                           const hipblasComplex* alpha,
                                           const hipblasComplex* x,
                                           int                   incx,
                                           hipblasStride         stridex,
                                           const hipblasComplex* y,
                                           int                   incy,
                                           hipblasStride         stridey,
                                           hipblasComplex*       A,
                                           int                   lda,
                                           hipblasStride         strideA,
                                           int                   batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZher2StridedBatched(hipblasHandle_t             handle,
                                           hipblasFillMode_t           uplo,
                                           int                         n,
                                           const hipblasDoubleComplex* alpha,
                                           const hipblasDoubleComplex* x,
                                           int                         incx,
                                           hipblasStride               stridex,
                                           const hipblasDoubleComplex* y,
                                           int                         incy,
                                           hipblasStride               stridey,
                                           hipblasDoubleComplex*       A,
                                           int                         lda,
                                           hipblasStride               strideA,
                                           int                         batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

// Level-2 : hpmv(supported datatypes : float complex and doule complex)
hipblasStatus_t hipblasChpmv(hipblasHandle_t handle, hipblasFillMode_t uplo, int n,
                            const hipblasComplex* alpha, const hipblasComplex* AP,
                            const hipblasComplex* x, int incx, const hipblasComplex* beta,
                            hipblasComplex* y, int incy)
try
{
    if (handle == nullptr || alpha == nullptr || AP == nullptr || x == nullptr || beta == nullptr || y == nullptr ||
        n <= 0 ) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    auto is_dev_ptr = (queryCurrentPtrMode(handle) == HIPBLAS_POINTER_MODE_DEVICE);

    float _Complex h_alpha, h_beta;
    if (is_dev_ptr) {
        hipMemcpy(&h_alpha, alpha, sizeof(float _Complex), hipMemcpyDefault);
        hipMemcpy(&h_beta, beta, sizeof(float _Complex), hipMemcpyDefault);
    } else {
        h_alpha = *((float _Complex*)alpha);
        h_beta = *((float _Complex*)beta);
    }

    onemklChpmv(sycl_queue, convert(uplo), n, h_alpha, (const float _Complex*)AP, (const float _Complex*)x, incx,
                h_beta, (float _Complex*)y, incy);
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZhpmv(hipblasHandle_t handle, hipblasFillMode_t uplo, int n,
                            const hipblasDoubleComplex* alpha, const hipblasDoubleComplex* AP,
                            const hipblasDoubleComplex* x, int incx, const hipblasDoubleComplex* beta,
                            hipblasDoubleComplex* y, int incy)
try
{
    if (handle == nullptr || alpha == nullptr || AP == nullptr || x == nullptr || beta == nullptr || y == nullptr ||
        n <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    auto is_dev_ptr = (queryCurrentPtrMode(handle) == HIPBLAS_POINTER_MODE_DEVICE);

    double _Complex h_alpha, h_beta;
    if (is_dev_ptr) {
        hipMemcpy(&h_alpha, alpha, sizeof(double _Complex), hipMemcpyDefault);
        hipMemcpy(&h_beta, beta, sizeof(double _Complex), hipMemcpyDefault);
    } else {
        h_alpha = *((double _Complex*)alpha);
        h_beta = *((double _Complex*)beta);
    }

    onemklZhpmv(sycl_queue, convert(uplo), n, h_alpha, (const double _Complex*)AP, (const double _Complex*)x, incx,
                h_beta, (double _Complex*)y, incy);
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}
// hpmv_batched
hipblasStatus_t hipblasChpmvBatched(hipblasHandle_t             handle,
                                    hipblasFillMode_t           uplo,
                                    int                         n,
                                    const hipblasComplex*       alpha,
                                    const hipblasComplex* const AP[],
                                    const hipblasComplex* const x[],
                                    int                         incx,
                                    const hipblasComplex*       beta,
                                    hipblasComplex* const       y[],
                                    int                         incy,
                                    int                         batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZhpmvBatched(hipblasHandle_t                   handle,
                                    hipblasFillMode_t                 uplo,
                                    int                               n,
                                    const hipblasDoubleComplex*       alpha,
                                    const hipblasDoubleComplex* const AP[],
                                    const hipblasDoubleComplex* const x[],
                                    int                               incx,
                                    const hipblasDoubleComplex*       beta,
                                    hipblasDoubleComplex* const       y[],
                                    int                               incy,
                                    int                               batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

// hpmv_strided_batched
hipblasStatus_t hipblasChpmvStridedBatched(hipblasHandle_t       handle,
                                           hipblasFillMode_t     uplo,
                                           int                   n,
                                           const hipblasComplex* alpha,
                                           const hipblasComplex* AP,
                                           hipblasStride         strideAP,
                                           const hipblasComplex* x,
                                           int                   incx,
                                           hipblasStride         stridex,
                                           const hipblasComplex* beta,
                                           hipblasComplex*       y,
                                           int                   incy,
                                           hipblasStride         stridey,
                                           int                   batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZhpmvStridedBatched(hipblasHandle_t             handle,
                                           hipblasFillMode_t           uplo,
                                           int                         n,
                                           const hipblasDoubleComplex* alpha,
                                           const hipblasDoubleComplex* AP,
                                           hipblasStride               strideAP,
                                           const hipblasDoubleComplex* x,
                                           int                         incx,
                                           hipblasStride               stridex,
                                           const hipblasDoubleComplex* beta,
                                           hipblasDoubleComplex*       y,
                                           int                         incy,
                                           hipblasStride               stridey,
                                           int                         batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}
// Level-2 : hpr(supported datatypes : float complex and doule complex)
hipblasStatus_t hipblasChpr(hipblasHandle_t       handle,
                            hipblasFillMode_t     uplo,
                            int                   n,
                            const float*          alpha,
                            const hipblasComplex* x,
                            int                   incx,
                            hipblasComplex*       AP)
try
{
    if (handle == nullptr || alpha == nullptr || AP == nullptr || x == nullptr ||
        n <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }

    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    // As per spec alpha can be device/host memory. In case of device memory *alpha will crash
    auto is_dev_ptr = (queryCurrentPtrMode(handle) == HIPBLAS_POINTER_MODE_DEVICE);

    float h_alpha;
    if (is_dev_ptr) {
        hipMemcpy(&h_alpha, alpha, sizeof(float), hipMemcpyDefault);
    } else {
        h_alpha = *((float*)alpha);
    }
    // workaround:
    // {AP := AP + alpha*x*x**H} -> for alpha=0 AP's value should not change
    // but incase of oneMKL it changes to zero. Once bug is fixed workaround should be removed
    if (h_alpha == 0) {
        return HIPBLAS_STATUS_SUCCESS;
    }
    onemklChpr(sycl_queue, convert(uplo), n, h_alpha, (const float _Complex*)x, incx, (float _Complex*)AP);
	return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZhpr(hipblasHandle_t             handle,
                            hipblasFillMode_t           uplo,
                            int                         n,
                            const double*               alpha,
                            const hipblasDoubleComplex* x,
                            int                         incx,
                            hipblasDoubleComplex*       AP)
try
{
    if (handle == nullptr || alpha == nullptr || AP == nullptr || x == nullptr ||
        n <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }

    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    // As per spec alpha can be device/host memory. In case of device memory *alpha will crash
    auto is_dev_ptr = (queryCurrentPtrMode(handle) == HIPBLAS_POINTER_MODE_DEVICE);

    double h_alpha;
    if (is_dev_ptr) {
        hipMemcpy(&h_alpha, alpha, sizeof(double), hipMemcpyDefault);
    } else {
        h_alpha = *((double*)alpha);
    }
    // workaround:
    // {AP := AP + alpha*x*x**H} -> for alpha=0 AP's value should not change
    // but incase of oneMKL it changes to zero. Once bug is fixed workaround should be removed
    if (h_alpha == 0) {
        return HIPBLAS_STATUS_SUCCESS;
    }
    onemklZhpr(sycl_queue, convert(uplo), n, h_alpha, (const double _Complex*)x, incx, (double _Complex*)AP);
	return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

// hpr_batched
hipblasStatus_t hipblasChprBatched(hipblasHandle_t             handle,
                                   hipblasFillMode_t           uplo,
                                   int                         n,
                                   const float*                alpha,
                                   const hipblasComplex* const x[],
                                   int                         incx,
                                   hipblasComplex* const       AP[],
                                   int                         batchCount)
try
{
	return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZhprBatched(hipblasHandle_t                   handle,
                                   hipblasFillMode_t                 uplo,
                                   int                               n,
                                   const double*                     alpha,
                                   const hipblasDoubleComplex* const x[],
                                   int                               incx,
                                   hipblasDoubleComplex* const       AP[],
                                   int                               batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

// hpr_strided_batched
hipblasStatus_t hipblasChprStridedBatched(hipblasHandle_t       handle,
                                          hipblasFillMode_t     uplo,
                                          int                   n,
                                          const float*          alpha,
                                          const hipblasComplex* x,
                                          int                   incx,
                                          hipblasStride         stridex,
                                          hipblasComplex*       AP,
                                          hipblasStride         strideAP,
                                          int                   batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZhprStridedBatched(hipblasHandle_t             handle,
                                          hipblasFillMode_t           uplo,
                                          int                         n,
                                          const double*               alpha,
                                          const hipblasDoubleComplex* x,
                                          int                         incx,
                                          hipblasStride               stridex,
                                          hipblasDoubleComplex*       AP,
                                          hipblasStride               strideAP,
                                          int                         batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

// Level-2 : hpr2(supported datatypes : float complex and doule complex)
hipblasStatus_t hipblasChpr2(hipblasHandle_t       handle,
                             hipblasFillMode_t     uplo,
                             int                   n,
                             const hipblasComplex* alpha,
                             const hipblasComplex* x,
                             int                   incx,
                             const hipblasComplex* y,
                             int                   incy,
                             hipblasComplex*       AP)
try
{
    if (handle == nullptr || alpha == nullptr || AP == nullptr || x == nullptr || y == nullptr ||
        n <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    // As per spec alpha can be device/host memory. In case of device memory *alpha will crash
    auto is_dev_ptr = (queryCurrentPtrMode(handle) == HIPBLAS_POINTER_MODE_DEVICE);

    float _Complex h_alpha;
    if (is_dev_ptr) {
        hipMemcpy(&h_alpha, alpha, sizeof(float _Complex), hipMemcpyDefault);
    } else {
        h_alpha = *((float _Complex*)alpha);
    }
    // workaround:
    // {AP := AP + alpha*x*y**H + conj(alpha)*y*x**H} -> for alpha=0 AP's value should not change
    // but incase of oneMKL it changes to zero. Once bug is fixed workaround should be removed
    if (h_alpha == 0) {
        return HIPBLAS_STATUS_SUCCESS;
    }
    onemklChpr2(sycl_queue, convert(uplo), n, h_alpha, (const float _Complex*)x, incx,
                            (const float _Complex*)y, incy, (float _Complex*)AP);
	return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZhpr2(hipblasHandle_t             handle,
                             hipblasFillMode_t           uplo,
                             int                         n,
                             const hipblasDoubleComplex* alpha,
                             const hipblasDoubleComplex* x,
                             int                         incx,
                             const hipblasDoubleComplex* y,
                             int                         incy,
                             hipblasDoubleComplex*       AP)
try
{
    if (handle == nullptr || alpha == nullptr || AP == nullptr || x == nullptr || y == nullptr ||
        n <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    // As per spec alpha can be device/host memory. In case of device memory *alpha will crash
    auto is_dev_ptr = (queryCurrentPtrMode(handle) == HIPBLAS_POINTER_MODE_DEVICE);

    double _Complex h_alpha;
    if (is_dev_ptr) {
        hipMemcpy(&h_alpha, alpha, sizeof(double _Complex), hipMemcpyDefault);
    } else {
        h_alpha = *((double _Complex*)alpha);
    }
    // workaround:
    // {AP := AP + alpha*x*y**H + conj(alpha)*y*x**H} -> for alpha=0 AP's value should not change
    // but incase of oneMKL it changes to zero. Once bug is fixed workaround should be removed
    if (h_alpha == 0) {
        return HIPBLAS_STATUS_SUCCESS;
    }
    onemklZhpr2(sycl_queue, convert(uplo), n, h_alpha, (const double _Complex*)x, incx,
                            (const double _Complex*)y, incy, (double _Complex*)AP);
	return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

// hpr2_batched
hipblasStatus_t hipblasChpr2Batched(hipblasHandle_t             handle,
                                    hipblasFillMode_t           uplo,
                                    int                         n,
                                    const hipblasComplex*       alpha,
                                    const hipblasComplex* const x[],
                                    int                         incx,
                                    const hipblasComplex* const y[],
                                    int                         incy,
                                    hipblasComplex* const       AP[],
                                    int                         batchCount)
try
{
	return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZhpr2Batched(hipblasHandle_t                   handle,
                                    hipblasFillMode_t                 uplo,
                                    int                               n,
                                    const hipblasDoubleComplex*       alpha,
                                    const hipblasDoubleComplex* const x[],
                                    int                               incx,
                                    const hipblasDoubleComplex* const y[],
                                    int                               incy,
                                    hipblasDoubleComplex* const       AP[],
                                    int                               batchCount)
try
{
	return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

// hpr2_strided_batched
hipblasStatus_t hipblasChpr2StridedBatched(hipblasHandle_t       handle,
                                           hipblasFillMode_t     uplo,
                                           int                   n,
                                           const hipblasComplex* alpha,
                                           const hipblasComplex* x,
                                           int                   incx,
                                           hipblasStride         stridex,
                                           const hipblasComplex* y,
                                           int                   incy,
                                           hipblasStride         stridey,
                                           hipblasComplex*       AP,
                                           hipblasStride         strideAP,
                                           int                   batchCount)
try
{
	return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZhpr2StridedBatched(hipblasHandle_t             handle,
                                           hipblasFillMode_t           uplo,
                                           int                         n,
                                           const hipblasDoubleComplex* alpha,
                                           const hipblasDoubleComplex* x,
                                           int                         incx,
                                           hipblasStride               stridex,
                                           const hipblasDoubleComplex* y,
                                           int                         incy,
                                           hipblasStride               stridey,
                                           hipblasDoubleComplex*       AP,
                                           hipblasStride               strideAP,
                                           int                         batchCount)
try
{
	return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

// Level-2 : sbmv(supported datatypes : float and doule )
hipblasStatus_t hipblasSsbmv(hipblasHandle_t   handle,
                             hipblasFillMode_t uplo,
                             int               n,
                             int               k,
                             const float*      alpha,
                             const float*      A,
                             int               lda,
                             const float*      x,
                             int               incx,
                             const float*      beta,
                             float*            y,
                             int               incy)
try
{
    if (handle == nullptr || alpha == nullptr || A == nullptr || x == nullptr || beta == nullptr || y == nullptr ||
        k <= 0 || n <= 0 || lda <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    // As per spec alpha can be device/host memory. In case of device memory *alpha will crash
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    auto is_dev_ptr = (queryCurrentPtrMode(handle) == HIPBLAS_POINTER_MODE_DEVICE);

    float h_alpha, h_beta;
    if (is_dev_ptr) {
        hipMemcpy(&h_alpha, alpha, sizeof(float), hipMemcpyDefault);
        hipMemcpy(&h_beta, beta, sizeof(float), hipMemcpyDefault);
    } else {
        h_alpha = *((float*)alpha);
        h_beta = *((float*)beta);
    }

    onemklSsbmv(sycl_queue, convert(uplo), n, k, h_alpha, A, lda, x, incx, h_beta, y, incy);
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDsbmv(hipblasHandle_t   handle,
                             hipblasFillMode_t uplo,
                             int               n,
                             int               k,
                             const double*     alpha,
                             const double*     A,
                             int               lda,
                             const double*     x,
                             int               incx,
                             const double*     beta,
                             double*           y,
                             int               incy)
try
{
    if (handle == nullptr || alpha == nullptr || A == nullptr || x == nullptr || beta == nullptr || y == nullptr ||
        k <= 0 || n <= 0 || lda <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    // As per spec alpha can be device/host memory. In case of device memory *alpha will crash
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    auto is_dev_ptr = (queryCurrentPtrMode(handle) == HIPBLAS_POINTER_MODE_DEVICE);

    double h_alpha, h_beta;
    if (is_dev_ptr) {
        hipMemcpy(&h_alpha, alpha, sizeof(double), hipMemcpyDefault);
        hipMemcpy(&h_beta, beta, sizeof(double), hipMemcpyDefault);
    } else {
        h_alpha = *((double*)alpha);
        h_beta = *((double*)beta);
    }

    onemklDsbmv(sycl_queue, convert(uplo), n, k, h_alpha, A, lda, x, incx, h_beta, y, incy);
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

// sbmv_batched
hipblasStatus_t hipblasSsbmvBatched(hipblasHandle_t    handle,
                                    hipblasFillMode_t  uplo,
                                    int                n,
                                    int                k,
                                    const float*       alpha,
                                    const float* const A[],
                                    int                lda,
                                    const float* const x[],
                                    int                incx,
                                    const float*       beta,
                                    float*             y[],
                                    int                incy,
                                    int                batchCount)
try
{
	return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDsbmvBatched(hipblasHandle_t     handle,
                                    hipblasFillMode_t   uplo,
                                    int                 n,
                                    int                 k,
                                    const double*       alpha,
                                    const double* const A[],
                                    int                 lda,
                                    const double* const x[],
                                    int                 incx,
                                    const double*       beta,
                                    double*             y[],
                                    int                 incy,
                                    int                 batchCount)
try
{
	return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

// sbmv_strided_batched
hipblasStatus_t hipblasSsbmvStridedBatched(hipblasHandle_t   handle,
                                           hipblasFillMode_t uplo,
                                           int               n,
                                           int               k,
                                           const float*      alpha,
                                           const float*      A,
                                           int               lda,
                                           hipblasStride     strideA,
                                           const float*      x,
                                           int               incx,
                                           hipblasStride     stridex,
                                           const float*      beta,
                                           float*            y,
                                           int               incy,
                                           hipblasStride     stridey,
                                           int               batchCount)
try
{
	return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDsbmvStridedBatched(hipblasHandle_t   handle,
                                           hipblasFillMode_t uplo,
                                           int               n,
                                           int               k,
                                           const double*     alpha,
                                           const double*     A,
                                           int               lda,
                                           hipblasStride     strideA,
                                           const double*     x,
                                           int               incx,
                                           hipblasStride     stridex,
                                           const double*     beta,
                                           double*           y,
                                           int               incy,
                                           hipblasStride     stridey,
                                           int               batchCount)
try
{
	return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

// Level-2 : spmv(supported datatypes : float and doule )
hipblasStatus_t hipblasSspmv(hipblasHandle_t   handle,
                             hipblasFillMode_t uplo,
                             int               n,
                             const float*      alpha,
                             const float*      AP,
                             const float*      x,
                             int               incx,
                             const float*      beta,
                             float*            y,
                             int               incy)
try
{
    if (handle == nullptr || alpha == nullptr || AP == nullptr || x == nullptr || beta == nullptr || y == nullptr ||
        n <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    // As per spec alpha can be device/host memory. In case of device memory *alpha will crash
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    auto is_dev_ptr = (queryCurrentPtrMode(handle) == HIPBLAS_POINTER_MODE_DEVICE);

    float h_alpha, h_beta;
    if (is_dev_ptr) {
        hipMemcpy(&h_alpha, alpha, sizeof(float), hipMemcpyDefault);
        hipMemcpy(&h_beta, beta, sizeof(float), hipMemcpyDefault);
    } else {
        h_alpha = *((float*)alpha);
        h_beta = *((float*)beta);
    }

    onemklSspmv(sycl_queue, convert(uplo), n, h_alpha, AP, x, incx, h_beta, y, incy);
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDspmv(hipblasHandle_t   handle,
                             hipblasFillMode_t uplo,
                             int               n,
                             const double*     alpha,
                             const double*     AP,
                             const double*     x,
                             int               incx,
                             const double*     beta,
                             double*           y,
                             int               incy)
try
{
    if (handle == nullptr || alpha == nullptr || AP == nullptr || x == nullptr || beta == nullptr || y == nullptr ||
        n <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    // As per spec alpha can be device/host memory. In case of device memory *alpha will crash
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    auto is_dev_ptr = (queryCurrentPtrMode(handle) == HIPBLAS_POINTER_MODE_DEVICE);

    double h_alpha, h_beta;
    if (is_dev_ptr) {
        hipMemcpy(&h_alpha, alpha, sizeof(double), hipMemcpyDefault);
        hipMemcpy(&h_beta, beta, sizeof(double), hipMemcpyDefault);
    } else {
        h_alpha = *((double*)alpha);
        h_beta = *((double*)beta);
    }

    onemklDspmv(sycl_queue, convert(uplo), n, h_alpha, AP, x, incx, h_beta, y, incy);
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

// spmv_batched
hipblasStatus_t hipblasSspmvBatched(hipblasHandle_t    handle,
                                    hipblasFillMode_t  uplo,
                                    int                n,
                                    const float*       alpha,
                                    const float* const AP[],
                                    const float* const x[],
                                    int                incx,
                                    const float*       beta,
                                    float*             y[],
                                    int                incy,
                                    int                batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDspmvBatched(hipblasHandle_t     handle,
                                    hipblasFillMode_t   uplo,
                                    int                 n,
                                    const double*       alpha,
                                    const double* const AP[],
                                    const double* const x[],
                                    int                 incx,
                                    const double*       beta,
                                    double*             y[],
                                    int                 incy,
                                    int                 batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

// spmv_strided_batched
hipblasStatus_t hipblasSspmvStridedBatched(hipblasHandle_t   handle,
                                           hipblasFillMode_t uplo,
                                           int               n,
                                           const float*      alpha,
                                           const float*      AP,
                                           hipblasStride     strideAP,
                                           const float*      x,
                                           int               incx,
                                           hipblasStride     stridex,
                                           const float*      beta,
                                           float*            y,
                                           int               incy,
                                           hipblasStride     stridey,
                                           int               batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDspmvStridedBatched(hipblasHandle_t   handle,
                                           hipblasFillMode_t uplo,
                                           int               n,
                                           const double*     alpha,
                                           const double*     AP,
                                           hipblasStride     strideAP,
                                           const double*     x,
                                           int               incx,
                                           hipblasStride     stridex,
                                           const double*     beta,
                                           double*           y,
                                           int               incy,
                                           hipblasStride     stridey,
                                           int               batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

// Level-2 : spr(supported datatypes : float, double, float complex and doule complex )
hipblasStatus_t hipblasSspr(hipblasHandle_t   handle,
                            hipblasFillMode_t uplo,
                            int               n,
                            const float*      alpha,
                            const float*      x,
                            int               incx,
                            float*            AP)
try
{
    if (handle == nullptr || alpha == nullptr || AP == nullptr || x == nullptr ||
        n <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    // As per spec alpha can be device/host memory. In case of device memory *alpha will crash
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    auto is_dev_ptr = (queryCurrentPtrMode(handle) == HIPBLAS_POINTER_MODE_DEVICE);

    float h_alpha;
    if (is_dev_ptr) {
        hipMemcpy(&h_alpha, alpha, sizeof(float), hipMemcpyDefault);
    } else {
        h_alpha = *((float*)alpha);
    }
    onemklSspr(sycl_queue, convert(uplo), n, h_alpha, x, incx, AP);
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDspr(hipblasHandle_t   handle,
                            hipblasFillMode_t uplo,
                            int               n,
                            const double*     alpha,
                            const double*     x,
                            int               incx,
                            double*           AP)
try
{
    if (handle == nullptr || alpha == nullptr || AP == nullptr || x == nullptr ||
        n <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    // As per spec alpha can be device/host memory. In case of device memory *alpha will crash
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    auto is_dev_ptr = (queryCurrentPtrMode(handle) == HIPBLAS_POINTER_MODE_DEVICE);

    double h_alpha;
    if (is_dev_ptr) {
        hipMemcpy(&h_alpha, alpha, sizeof(double), hipMemcpyDefault);
    } else {
        h_alpha = *((double*)alpha);
    }
    onemklDspr(sycl_queue, convert(uplo), n, h_alpha, x, incx, AP);
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCspr(hipblasHandle_t       handle,
                            hipblasFillMode_t     uplo,
                            int                   n,
                            const hipblasComplex* alpha,
                            const hipblasComplex* x,
                            int                   incx,
                            hipblasComplex*       AP)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZspr(hipblasHandle_t             handle,
                            hipblasFillMode_t           uplo,
                            int                         n,
                            const hipblasDoubleComplex* alpha,
                            const hipblasDoubleComplex* x,
                            int                         incx,
                            hipblasDoubleComplex*       AP)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

// spr_batched
hipblasStatus_t hipblasSsprBatched(hipblasHandle_t    handle,
                                   hipblasFillMode_t  uplo,
                                   int                n,
                                   const float*       alpha,
                                   const float* const x[],
                                   int                incx,
                                   float* const       AP[],
                                   int                batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDsprBatched(hipblasHandle_t     handle,
                                   hipblasFillMode_t   uplo,
                                   int                 n,
                                   const double*       alpha,
                                   const double* const x[],
                                   int                 incx,
                                   double* const       AP[],
                                   int                 batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCsprBatched(hipblasHandle_t             handle,
                                   hipblasFillMode_t           uplo,
                                   int                         n,
                                   const hipblasComplex*       alpha,
                                   const hipblasComplex* const x[],
                                   int                         incx,
                                   hipblasComplex* const       AP[],
                                   int                         batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZsprBatched(hipblasHandle_t                   handle,
                                   hipblasFillMode_t                 uplo,
                                   int                               n,
                                   const hipblasDoubleComplex*       alpha,
                                   const hipblasDoubleComplex* const x[],
                                   int                               incx,
                                   hipblasDoubleComplex* const       AP[],
                                   int                               batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

// spr_strided_batched
hipblasStatus_t hipblasSsprStridedBatched(hipblasHandle_t   handle,
                                          hipblasFillMode_t uplo,
                                          int               n,
                                          const float*      alpha,
                                          const float*      x,
                                          int               incx,
                                          hipblasStride     stridex,
                                          float*            AP,
                                          hipblasStride     strideAP,
                                          int               batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDsprStridedBatched(hipblasHandle_t   handle,
                                          hipblasFillMode_t uplo,
                                          int               n,
                                          const double*     alpha,
                                          const double*     x,
                                          int               incx,
                                          hipblasStride     stridex,
                                          double*           AP,
                                          hipblasStride     strideAP,
                                          int               batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCsprStridedBatched(hipblasHandle_t       handle,
                                          hipblasFillMode_t     uplo,
                                          int                   n,
                                          const hipblasComplex* alpha,
                                          const hipblasComplex* x,
                                          int                   incx,
                                          hipblasStride         stridex,
                                          hipblasComplex*       AP,
                                          hipblasStride         strideAP,
                                          int                   batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZsprStridedBatched(hipblasHandle_t             handle,
                                          hipblasFillMode_t           uplo,
                                          int                         n,
                                          const hipblasDoubleComplex* alpha,
                                          const hipblasDoubleComplex* x,
                                          int                         incx,
                                          hipblasStride               stridex,
                                          hipblasDoubleComplex*       AP,
                                          hipblasStride               strideAP,
                                          int                         batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

// Level-2 : spr2(supported datatypes : float and double )
hipblasStatus_t hipblasSspr2(hipblasHandle_t   handle,
                             hipblasFillMode_t uplo,
                             int               n,
                             const float*      alpha,
                             const float*      x,
                             int               incx,
                             const float*      y,
                             int               incy,
                             float*            AP)
try
{
    if (handle == nullptr || alpha == nullptr || AP == nullptr || x == nullptr || y == nullptr ||
        n <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    // As per spec alpha can be device/host memory. In case of device memory *alpha will crash
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    auto is_dev_ptr = (queryCurrentPtrMode(handle) == HIPBLAS_POINTER_MODE_DEVICE);

    float h_alpha;
    if (is_dev_ptr) {
        hipMemcpy(&h_alpha, alpha, sizeof(float), hipMemcpyDefault);
    } else {
        h_alpha = *((float*)alpha);
    }
    onemklSspr2(sycl_queue, convert(uplo), n, h_alpha, x, incx, y, incy, AP);
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDspr2(hipblasHandle_t   handle,
                             hipblasFillMode_t uplo,
                             int               n,
                             const double*     alpha,
                             const double*     x,
                             int               incx,
                             const double*     y,
                             int               incy,
                             double*           AP)
try
{
    if (handle == nullptr || alpha == nullptr || AP == nullptr || x == nullptr || y == nullptr ||
        n <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    // As per spec alpha can be device/host memory. In case of device memory *alpha will crash
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    auto is_dev_ptr = (queryCurrentPtrMode(handle) == HIPBLAS_POINTER_MODE_DEVICE);

    double h_alpha;
    if (is_dev_ptr) {
        hipMemcpy(&h_alpha, alpha, sizeof(double), hipMemcpyDefault);
    } else {
        h_alpha = *((double*)alpha);
    }
    onemklDspr2(sycl_queue, convert(uplo), n, h_alpha, x, incx, y, incy, AP);
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

// spr2_batched
hipblasStatus_t hipblasSspr2Batched(hipblasHandle_t    handle,
                                    hipblasFillMode_t  uplo,
                                    int                n,
                                    const float*       alpha,
                                    const float* const x[],
                                    int                incx,
                                    const float* const y[],
                                    int                incy,
                                    float* const       AP[],
                                    int                batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDspr2Batched(hipblasHandle_t     handle,
                                    hipblasFillMode_t   uplo,
                                    int                 n,
                                    const double*       alpha,
                                    const double* const x[],
                                    int                 incx,
                                    const double* const y[],
                                    int                 incy,
                                    double* const       AP[],
                                    int                 batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

// spr2_strided_batched
hipblasStatus_t hipblasSspr2StridedBatched(hipblasHandle_t   handle,
                                           hipblasFillMode_t uplo,
                                           int               n,
                                           const float*      alpha,
                                           const float*      x,
                                           int               incx,
                                           hipblasStride     stridex,
                                           const float*      y,
                                           int               incy,
                                           hipblasStride     stridey,
                                           float*            AP,
                                           hipblasStride     strideAP,
                                           int               batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDspr2StridedBatched(hipblasHandle_t   handle,
                                           hipblasFillMode_t uplo,
                                           int               n,
                                           const double*     alpha,
                                           const double*     x,
                                           int               incx,
                                           hipblasStride     stridex,
                                           const double*     y,
                                           int               incy,
                                           hipblasStride     stridey,
                                           double*           AP,
                                           hipblasStride     strideAP,
                                           int               batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

// Level-2 : symv(supported datatypes : float and double )
hipblasStatus_t hipblasSsymv(hipblasHandle_t   handle,
                             hipblasFillMode_t uplo,
                             int               n,
                             const float*      alpha,
                             const float*      A,
                             int               lda,
                             const float*      x,
                             int               incx,
                             const float*      beta,
                             float*            y,
                             int               incy)
try
{
    if (handle == nullptr || alpha == nullptr || A == nullptr || x == nullptr || y == nullptr || beta == nullptr ||
        n <= 0 || lda <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }

    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    auto is_dev_ptr = (queryCurrentPtrMode(handle) == HIPBLAS_POINTER_MODE_DEVICE);

    // As per spec alpha can be device/host memory. In case of device memory *alpha will crash
    float h_alpha, h_beta;
    if (is_dev_ptr) {
        hipMemcpy(&h_alpha, alpha, sizeof(float), hipMemcpyDefault);
        hipMemcpy(&h_beta, beta, sizeof(float), hipMemcpyDefault);
    } else {
        h_alpha = *((float*)alpha);
        h_beta = *((float*)beta);
    }
    onemklSsymv(sycl_queue, convert(uplo), n, h_alpha, A, lda, x, incx, h_beta, y, incy);
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDsymv(hipblasHandle_t   handle,
                             hipblasFillMode_t uplo,
                             int               n,
                             const double*     alpha,
                             const double*     A,
                             int               lda,
                             const double*     x,
                             int               incx,
                             const double*     beta,
                             double*           y,
                             int               incy)
try
{
    if (handle == nullptr || alpha == nullptr || A == nullptr || x == nullptr || y == nullptr || beta == nullptr ||
        n <= 0 || lda <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }

    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    auto is_dev_ptr = (queryCurrentPtrMode(handle) == HIPBLAS_POINTER_MODE_DEVICE);

    // As per spec alpha can be device/host memory. In case of device memory *alpha will crash
    double h_alpha, h_beta;
    if (is_dev_ptr) {
        hipMemcpy(&h_alpha, alpha, sizeof(double), hipMemcpyDefault);
        hipMemcpy(&h_beta, beta, sizeof(double), hipMemcpyDefault);
    } else {
        h_alpha = *((double*)alpha);
        h_beta = *((double*)beta);
    }

    onemklDsymv(sycl_queue, convert(uplo), n, h_alpha, A, lda, x, incx, h_beta, y, incy);
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCsymv(hipblasHandle_t       handle,
                             hipblasFillMode_t     uplo,
                             int                   n,
                             const hipblasComplex* alpha,
                             const hipblasComplex* A,
                             int                   lda,
                             const hipblasComplex* x,
                             int                   incx,
                             const hipblasComplex* beta,
                             hipblasComplex*       y,
                             int                   incy)
try
{
    if (handle == nullptr || alpha == nullptr || A == nullptr || x == nullptr || y == nullptr || beta == nullptr ||
        n <= 0 || lda <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    auto is_dev_ptr = (queryCurrentPtrMode(handle) == HIPBLAS_POINTER_MODE_DEVICE);
    // As per spec alpha can be device/host memory. In case of device memory *alpha will crash
    float _Complex h_alpha, h_beta;
    if (is_dev_ptr) {
        hipMemcpy(&h_alpha, alpha, sizeof(float _Complex), hipMemcpyDefault);
        hipMemcpy(&h_beta, beta, sizeof(float _Complex), hipMemcpyDefault);
    } else {
        h_alpha = *((float _Complex*)alpha);
        h_beta = *((float _Complex*)beta);
    }
    // call needs to add
    onemklCsymv(sycl_queue, convert(uplo), n, h_alpha, (const float _Complex*)A, lda,
                (const float _Complex*)x, incx, h_beta,
                (float _Complex*)y, incy);
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZsymv(hipblasHandle_t             handle,
                             hipblasFillMode_t           uplo,
                             int                         n,
                             const hipblasDoubleComplex* alpha,
                             const hipblasDoubleComplex* A,
                             int                         lda,
                             const hipblasDoubleComplex* x,
                             int                         incx,
                             const hipblasDoubleComplex* beta,
                             hipblasDoubleComplex*       y,
                             int                         incy)
try
{
    if (handle == nullptr || alpha == nullptr || A == nullptr || x == nullptr || y == nullptr || beta == nullptr ||
        n <= 0 || lda <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    auto is_dev_ptr = (queryCurrentPtrMode(handle) == HIPBLAS_POINTER_MODE_DEVICE);
    // As per spec alpha can be device/host memory. In case of device memory *alpha will crash
    double _Complex h_alpha, h_beta;
    if (is_dev_ptr) {
        hipMemcpy(&h_alpha, alpha, sizeof(double _Complex), hipMemcpyDefault);
        hipMemcpy(&h_beta, beta, sizeof(double _Complex), hipMemcpyDefault);
    } else {
        h_alpha = *((double _Complex*)alpha);
        h_beta = *((double _Complex*)beta);
    }
    onemklZsymv(sycl_queue, convert(uplo), n, h_alpha, (const double _Complex*)A, lda,
                (const double _Complex*)x, incx, h_beta,
                (double _Complex*)y, incy);
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

// symv_batched
hipblasStatus_t hipblasSsymvBatched(hipblasHandle_t    handle,
                                    hipblasFillMode_t  uplo,
                                    int                n,
                                    const float*       alpha,
                                    const float* const A[],
                                    int                lda,
                                    const float* const x[],
                                    int                incx,
                                    const float*       beta,
                                    float*             y[],
                                    int                incy,
                                    int                batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDsymvBatched(hipblasHandle_t     handle,
                                    hipblasFillMode_t   uplo,
                                    int                 n,
                                    const double*       alpha,
                                    const double* const A[],
                                    int                 lda,
                                    const double* const x[],
                                    int                 incx,
                                    const double*       beta,
                                    double*             y[],
                                    int                 incy,
                                    int                 batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCsymvBatched(hipblasHandle_t             handle,
                                    hipblasFillMode_t           uplo,
                                    int                         n,
                                    const hipblasComplex*       alpha,
                                    const hipblasComplex* const A[],
                                    int                         lda,
                                    const hipblasComplex* const x[],
                                    int                         incx,
                                    const hipblasComplex*       beta,
                                    hipblasComplex*             y[],
                                    int                         incy,
                                    int                         batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZsymvBatched(hipblasHandle_t                   handle,
                                    hipblasFillMode_t                 uplo,
                                    int                               n,
                                    const hipblasDoubleComplex*       alpha,
                                    const hipblasDoubleComplex* const A[],
                                    int                               lda,
                                    const hipblasDoubleComplex* const x[],
                                    int                               incx,
                                    const hipblasDoubleComplex*       beta,
                                    hipblasDoubleComplex*             y[],
                                    int                               incy,
                                    int                               batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

// symv_strided_batched
hipblasStatus_t hipblasSsymvStridedBatched(hipblasHandle_t   handle,
                                           hipblasFillMode_t uplo,
                                           int               n,
                                           const float*      alpha,
                                           const float*      A,
                                           int               lda,
                                           hipblasStride     strideA,
                                           const float*      x,
                                           int               incx,
                                           hipblasStride     stridex,
                                           const float*      beta,
                                           float*            y,
                                           int               incy,
                                           hipblasStride     stridey,
                                           int               batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDsymvStridedBatched(hipblasHandle_t   handle,
                                           hipblasFillMode_t uplo,
                                           int               n,
                                           const double*     alpha,
                                           const double*     A,
                                           int               lda,
                                           hipblasStride     strideA,
                                           const double*     x,
                                           int               incx,
                                           hipblasStride     stridex,
                                           const double*     beta,
                                           double*           y,
                                           int               incy,
                                           hipblasStride     stridey,
                                           int               batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCsymvStridedBatched(hipblasHandle_t       handle,
                                           hipblasFillMode_t     uplo,
                                           int                   n,
                                           const hipblasComplex* alpha,
                                           const hipblasComplex* A,
                                           int                   lda,
                                           hipblasStride         strideA,
                                           const hipblasComplex* x,
                                           int                   incx,
                                           hipblasStride         stridex,
                                           const hipblasComplex* beta,
                                           hipblasComplex*       y,
                                           int                   incy,
                                           hipblasStride         stridey,
                                           int                   batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZsymvStridedBatched(hipblasHandle_t             handle,
                                           hipblasFillMode_t           uplo,
                                           int                         n,
                                           const hipblasDoubleComplex* alpha,
                                           const hipblasDoubleComplex* A,
                                           int                         lda,
                                           hipblasStride               strideA,
                                           const hipblasDoubleComplex* x,
                                           int                         incx,
                                           hipblasStride               stridex,
                                           const hipblasDoubleComplex* beta,
                                           hipblasDoubleComplex*       y,
                                           int                         incy,
                                           hipblasStride               stridey,
                                           int                         batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

// Level-2 : syr(supported datatypes : float and double )
hipblasStatus_t hipblasSsyr(hipblasHandle_t   handle,
                            hipblasFillMode_t uplo,
                            int               n,
                            const float*      alpha,
                            const float*      x,
                            int               incx,
                            float*            A,
                            int               lda)
try
{
    if (handle == nullptr || alpha == nullptr || A == nullptr || x == nullptr ||
        n <= 0 || incx <= 0 || lda <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    // As per spec alpha can be device/host memory. In case of device memory *alpha will crash
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    auto is_dev_ptr = (queryCurrentPtrMode(handle) == HIPBLAS_POINTER_MODE_DEVICE);

    float h_alpha;
    if (is_dev_ptr) {
        hipMemcpy(&h_alpha, alpha, sizeof(float), hipMemcpyDefault);
    } else {
        h_alpha = *((float*)alpha);
    }
    onemklSsyr(sycl_queue, convert(uplo), n, h_alpha, x, incx, A, lda);
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDsyr(hipblasHandle_t   handle,
                            hipblasFillMode_t uplo,
                            int               n,
                            const double*     alpha,
                            const double*     x,
                            int               incx,
                            double*           A,
                            int               lda)
try
{
    if (handle == nullptr || alpha == nullptr || A == nullptr || x == nullptr ||
        n <= 0 || incx <= 0 || lda <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    // As per spec alpha can be device/host memory. In case of device memory *alpha will crash
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    auto is_dev_ptr = (queryCurrentPtrMode(handle) == HIPBLAS_POINTER_MODE_DEVICE);

    double h_alpha;
    if (is_dev_ptr) {
        hipMemcpy(&h_alpha, alpha, sizeof(double), hipMemcpyDefault);
    } else {
        h_alpha = *((double*)alpha);
    }
    onemklDsyr(sycl_queue, convert(uplo), n, h_alpha, x, incx, A, lda);
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCsyr(hipblasHandle_t       handle,
                            hipblasFillMode_t     uplo,
                            int                   n,
                            const hipblasComplex* alpha,
                            const hipblasComplex* x,
                            int                   incx,
                            hipblasComplex*       A,
                            int                   lda)
try
{
    if (handle == nullptr || alpha == nullptr || A == nullptr || x == nullptr ||
        n <= 0 || incx <= 0 || lda <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }

    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    auto is_dev_ptr = (queryCurrentPtrMode(handle) == HIPBLAS_POINTER_MODE_DEVICE);

    // As per spec alpha can be device/host memory. In case of device memory *alpha will crash
    float _Complex h_alpha;
    if (is_dev_ptr) {
        hipMemcpy(&h_alpha, alpha, sizeof(float _Complex), hipMemcpyDefault);
    } else {
        h_alpha = *((float _Complex*)alpha);
    }
    onemklCsyr(sycl_queue, convert(uplo), n, h_alpha, (const float _Complex*)x, incx, (float _Complex*)A, lda);
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZsyr(hipblasHandle_t             handle,
                            hipblasFillMode_t           uplo,
                            int                         n,
                            const hipblasDoubleComplex* alpha,
                            const hipblasDoubleComplex* x,
                            int                         incx,
                            hipblasDoubleComplex*       A,
                            int                         lda)
try
{
    if (handle == nullptr || alpha == nullptr || A == nullptr || x == nullptr ||
        n <= 0 || incx <= 0 || lda <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }

    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    auto is_dev_ptr = (queryCurrentPtrMode(handle) == HIPBLAS_POINTER_MODE_DEVICE);

    // As per spec alpha can be device/host memory. In case of device memory *alpha will crash
    double _Complex h_alpha;
    if (is_dev_ptr) {
        hipMemcpy(&h_alpha, alpha, sizeof(double _Complex), hipMemcpyDefault);
    } else {
        h_alpha = *((double _Complex*)alpha);
    }
    onemklZsyr(sycl_queue, convert(uplo), n, h_alpha, (const double _Complex*)x, incx, (double _Complex*)A, lda);
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

// syr_batched
hipblasStatus_t hipblasSsyrBatched(hipblasHandle_t    handle,
                                   hipblasFillMode_t  uplo,
                                   int                n,
                                   const float*       alpha,
                                   const float* const x[],
                                   int                incx,
                                   float* const       A[],
                                   int                lda,
                                   int                batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDsyrBatched(hipblasHandle_t     handle,
                                   hipblasFillMode_t   uplo,
                                   int                 n,
                                   const double*       alpha,
                                   const double* const x[],
                                   int                 incx,
                                   double* const       A[],
                                   int                 lda,
                                   int                 batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCsyrBatched(hipblasHandle_t             handle,
                                   hipblasFillMode_t           uplo,
                                   int                         n,
                                   const hipblasComplex*       alpha,
                                   const hipblasComplex* const x[],
                                   int                         incx,
                                   hipblasComplex* const       A[],
                                   int                         lda,
                                   int                         batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZsyrBatched(hipblasHandle_t                   handle,
                                   hipblasFillMode_t                 uplo,
                                   int                               n,
                                   const hipblasDoubleComplex*       alpha,
                                   const hipblasDoubleComplex* const x[],
                                   int                               incx,
                                   hipblasDoubleComplex* const       A[],
                                   int                               lda,
                                   int                               batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

// syr_strided_batched
hipblasStatus_t hipblasSsyrStridedBatched(hipblasHandle_t   handle,
                                          hipblasFillMode_t uplo,
                                          int               n,
                                          const float*      alpha,
                                          const float*      x,
                                          int               incx,
                                          hipblasStride     stridex,
                                          float*            A,
                                          int               lda,
                                          hipblasStride     strideA,
                                          int               batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDsyrStridedBatched(hipblasHandle_t   handle,
                                          hipblasFillMode_t uplo,
                                          int               n,
                                          const double*     alpha,
                                          const double*     x,
                                          int               incx,
                                          hipblasStride     stridex,
                                          double*           A,
                                          int               lda,
                                          hipblasStride     strideA,
                                          int               batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCsyrStridedBatched(hipblasHandle_t       handle,
                                          hipblasFillMode_t     uplo,
                                          int                   n,
                                          const hipblasComplex* alpha,
                                          const hipblasComplex* x,
                                          int                   incx,
                                          hipblasStride         stridex,
                                          hipblasComplex*       A,
                                          int                   lda,
                                          hipblasStride         strideA,
                                          int                   batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZsyrStridedBatched(hipblasHandle_t             handle,
                                          hipblasFillMode_t           uplo,
                                          int                         n,
                                          const hipblasDoubleComplex* alpha,
                                          const hipblasDoubleComplex* x,
                                          int                         incx,
                                          hipblasStride               stridex,
                                          hipblasDoubleComplex*       A,
                                          int                         lda,
                                          hipblasStride               strideA,
                                          int                         batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

// Level-2 : syr2(supported datatypes : float and double )
hipblasStatus_t hipblasSsyr2(hipblasHandle_t   handle,
                             hipblasFillMode_t uplo,
                             int               n,
                             const float*      alpha,
                             const float*      x,
                             int               incx,
                             const float*      y,
                             int               incy,
                             float*            A,
                             int               lda)
try
{
    if (handle == nullptr || alpha == nullptr || A == nullptr || x == nullptr || y == nullptr ||
        n <= 0 || lda <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    // As per spec alpha can be device/host memory. In case of device memory *alpha will crash
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    auto is_dev_ptr = (queryCurrentPtrMode(handle) == HIPBLAS_POINTER_MODE_DEVICE);

    float h_alpha;
    if (is_dev_ptr) {
        hipMemcpy(&h_alpha, alpha, sizeof(float), hipMemcpyDefault);
    } else {
        h_alpha = *((float*)alpha);
    }
    onemklSsyr2(sycl_queue, convert(uplo), n, h_alpha, x, incx, y, incy, A, lda);
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDsyr2(hipblasHandle_t   handle,
                             hipblasFillMode_t uplo,
                             int               n,
                             const double*     alpha,
                             const double*     x,
                             int               incx,
                             const double*     y,
                             int               incy,
                             double*           A,
                             int               lda)
try
{
    if (handle == nullptr || alpha == nullptr || A == nullptr || x == nullptr || y == nullptr ||
        n <= 0 || lda <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    // As per spec alpha can be device/host memory. In case of device memory *alpha will crash
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    auto is_dev_ptr = (queryCurrentPtrMode(handle) == HIPBLAS_POINTER_MODE_DEVICE);

    double h_alpha;
    if (is_dev_ptr) {
        hipMemcpy(&h_alpha, alpha, sizeof(double), hipMemcpyDefault);
    } else {
        h_alpha = *((double*)alpha);
    }
    onemklDsyr2(sycl_queue, convert(uplo), n, h_alpha, x, incx, y, incy, A, lda);
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCsyr2(hipblasHandle_t       handle,
                             hipblasFillMode_t     uplo,
                             int                   n,
                             const hipblasComplex* alpha,
                             const hipblasComplex* x,
                             int                   incx,
                             const hipblasComplex* y,
                             int                   incy,
                             hipblasComplex*       A,
                             int                   lda)
try
{
    if (handle == nullptr || alpha == nullptr || A == nullptr || x == nullptr || y == nullptr ||
        n <= 0 || lda <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    // As per spec alpha can be device/host memory. In case of device memory *alpha will crash
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    auto is_dev_ptr = (queryCurrentPtrMode(handle) == HIPBLAS_POINTER_MODE_DEVICE);

    float _Complex h_alpha;
    if (is_dev_ptr) {
        hipMemcpy(&h_alpha, alpha, sizeof(float _Complex), hipMemcpyDefault);
    } else {
        h_alpha = *((float _Complex*)alpha);
    }
    onemklCsyr2(sycl_queue, convert(uplo), n, h_alpha, (const float _Complex*)x, incx,
                (const float _Complex*)y, incy, (float _Complex*)A, lda);
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZsyr2(hipblasHandle_t             handle,
                             hipblasFillMode_t           uplo,
                             int                         n,
                             const hipblasDoubleComplex* alpha,
                             const hipblasDoubleComplex* x,
                             int                         incx,
                             const hipblasDoubleComplex* y,
                             int                         incy,
                             hipblasDoubleComplex*       A,
                             int                         lda)
try
{
    if (handle == nullptr || alpha == nullptr || A == nullptr || x == nullptr || y == nullptr ||
        n <= 0 || lda <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    // As per spec alpha can be device/host memory. In case of device memory *alpha will crash
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    auto is_dev_ptr = (queryCurrentPtrMode(handle) == HIPBLAS_POINTER_MODE_DEVICE);

    double _Complex h_alpha;
    if (is_dev_ptr) {
        hipMemcpy(&h_alpha, alpha, sizeof(double _Complex), hipMemcpyDefault);
    } else {
        h_alpha = *((double _Complex*)alpha);
    }
    onemklZsyr2(sycl_queue, convert(uplo), n, h_alpha, (const double _Complex*)x, incx,
                (const double _Complex*)y, incy, (double _Complex*)A, lda);
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

// syr2_batched
hipblasStatus_t hipblasSsyr2Batched(hipblasHandle_t    handle,
                                    hipblasFillMode_t  uplo,
                                    int                n,
                                    const float*       alpha,
                                    const float* const x[],
                                    int                incx,
                                    const float* const y[],
                                    int                incy,
                                    float* const       A[],
                                    int                lda,
                                    int                batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDsyr2Batched(hipblasHandle_t     handle,
                                    hipblasFillMode_t   uplo,
                                    int                 n,
                                    const double*       alpha,
                                    const double* const x[],
                                    int                 incx,
                                    const double* const y[],
                                    int                 incy,
                                    double* const       A[],
                                    int                 lda,
                                    int                 batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCsyr2Batched(hipblasHandle_t             handle,
                                    hipblasFillMode_t           uplo,
                                    int                         n,
                                    const hipblasComplex*       alpha,
                                    const hipblasComplex* const x[],
                                    int                         incx,
                                    const hipblasComplex* const y[],
                                    int                         incy,
                                    hipblasComplex* const       A[],
                                    int                         lda,
                                    int                         batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZsyr2Batched(hipblasHandle_t                   handle,
                                    hipblasFillMode_t                 uplo,
                                    int                               n,
                                    const hipblasDoubleComplex*       alpha,
                                    const hipblasDoubleComplex* const x[],
                                    int                               incx,
                                    const hipblasDoubleComplex* const y[],
                                    int                               incy,
                                    hipblasDoubleComplex* const       A[],
                                    int                               lda,
                                    int                               batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

// syr2_strided_batched
hipblasStatus_t hipblasSsyr2StridedBatched(hipblasHandle_t   handle,
                                           hipblasFillMode_t uplo,
                                           int               n,
                                           const float*      alpha,
                                           const float*      x,
                                           int               incx,
                                           hipblasStride     stridex,
                                           const float*      y,
                                           int               incy,
                                           hipblasStride     stridey,
                                           float*            A,
                                           int               lda,
                                           hipblasStride     strideA,
                                           int               batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDsyr2StridedBatched(hipblasHandle_t   handle,
                                           hipblasFillMode_t uplo,
                                           int               n,
                                           const double*     alpha,
                                           const double*     x,
                                           int               incx,
                                           hipblasStride     stridex,
                                           const double*     y,
                                           int               incy,
                                           hipblasStride     stridey,
                                           double*           A,
                                           int               lda,
                                           hipblasStride     strideA,
                                           int               batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCsyr2StridedBatched(hipblasHandle_t       handle,
                                           hipblasFillMode_t     uplo,
                                           int                   n,
                                           const hipblasComplex* alpha,
                                           const hipblasComplex* x,
                                           int                   incx,
                                           hipblasStride         stridex,
                                           const hipblasComplex* y,
                                           int                   incy,
                                           hipblasStride         stridey,
                                           hipblasComplex*       A,
                                           int                   lda,
                                           hipblasStride         strideA,
                                           int                   batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZsyr2StridedBatched(hipblasHandle_t             handle,
                                           hipblasFillMode_t           uplo,
                                           int                         n,
                                           const hipblasDoubleComplex* alpha,
                                           const hipblasDoubleComplex* x,
                                           int                         incx,
                                           hipblasStride               stridex,
                                           const hipblasDoubleComplex* y,
                                           int                         incy,
                                           hipblasStride               stridey,
                                           hipblasDoubleComplex*       A,
                                           int                         lda,
                                           hipblasStride               strideA,
                                           int                         batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

// Level-2 : tbmv(supported datatypes : float , double , float complex and double complex )
hipblasStatus_t hipblasStbmv(hipblasHandle_t    handle,
                             hipblasFillMode_t  uplo,
                             hipblasOperation_t transA,
                             hipblasDiagType_t  diag,
                             int                m,
                             int                k,
                             const float*       A,
                             int                lda,
                             float*             x,
                             int                incx)
try
{
    if (handle == nullptr || A == nullptr || x == nullptr ||
        m <= 0 || k <= 0 || lda <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }

    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    onemklStbmv(sycl_queue, convert(uplo), convert(transA), convert(diag), m, k, A, lda, x, incx);
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDtbmv(hipblasHandle_t    handle,
                             hipblasFillMode_t  uplo,
                             hipblasOperation_t transA,
                             hipblasDiagType_t  diag,
                             int                m,
                             int                k,
                             const double*      A,
                             int                lda,
                             double*            x,
                             int                incx)
try
{
    if (handle == nullptr || A == nullptr || x == nullptr ||
        m <= 0 || k <= 0 || lda <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    onemklDtbmv(sycl_queue, convert(uplo), convert(transA), convert(diag), m, k, A, lda, x, incx);
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCtbmv(hipblasHandle_t       handle,
                             hipblasFillMode_t     uplo,
                             hipblasOperation_t    transA,
                             hipblasDiagType_t     diag,
                             int                   m,
                             int                   k,
                             const hipblasComplex* A,
                             int                   lda,
                             hipblasComplex*       x,
                             int                   incx)
try
{
    if (handle == nullptr || A == nullptr || x == nullptr ||
        m <= 0 || k <= 0 || lda <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    onemklCtbmv(sycl_queue, convert(uplo), convert(transA), convert(diag), m, k, 
                            (const float _Complex*)A, lda, (float _Complex*)x, incx);
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZtbmv(hipblasHandle_t             handle,
                             hipblasFillMode_t           uplo,
                             hipblasOperation_t          transA,
                             hipblasDiagType_t           diag,
                             int                         m,
                             int                         k,
                             const hipblasDoubleComplex* A,
                             int                         lda,
                             hipblasDoubleComplex*       x,
                             int                         incx)
try
{
    if (handle == nullptr || A == nullptr || x == nullptr ||
        m <= 0 || k <= 0 || lda <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    onemklZtbmv(sycl_queue, convert(uplo), convert(transA), convert(diag), m, k, 
                            (const double _Complex*)A, lda, (double _Complex*)x, incx);
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

// tbmv_batched
hipblasStatus_t hipblasStbmvBatched(hipblasHandle_t    handle,
                                    hipblasFillMode_t  uplo,
                                    hipblasOperation_t transA,
                                    hipblasDiagType_t  diag,
                                    int                m,
                                    int                k,
                                    const float* const A[],
                                    int                lda,
                                    float* const       x[],
                                    int                incx,
                                    int                batch_count)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDtbmvBatched(hipblasHandle_t     handle,
                                    hipblasFillMode_t   uplo,
                                    hipblasOperation_t  transA,
                                    hipblasDiagType_t   diag,
                                    int                 m,
                                    int                 k,
                                    const double* const A[],
                                    int                 lda,
                                    double* const       x[],
                                    int                 incx,
                                    int                 batch_count)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCtbmvBatched(hipblasHandle_t             handle,
                                    hipblasFillMode_t           uplo,
                                    hipblasOperation_t          transA,
                                    hipblasDiagType_t           diag,
                                    int                         m,
                                    int                         k,
                                    const hipblasComplex* const A[],
                                    int                         lda,
                                    hipblasComplex* const       x[],
                                    int                         incx,
                                    int                         batch_count)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZtbmvBatched(hipblasHandle_t                   handle,
                                    hipblasFillMode_t                 uplo,
                                    hipblasOperation_t                transA,
                                    hipblasDiagType_t                 diag,
                                    int                               m,
                                    int                               k,
                                    const hipblasDoubleComplex* const A[],
                                    int                               lda,
                                    hipblasDoubleComplex* const       x[],
                                    int                               incx,
                                    int                               batch_count)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

// tbmv_strided_batched
hipblasStatus_t hipblasStbmvStridedBatched(hipblasHandle_t    handle,
                                           hipblasFillMode_t  uplo,
                                           hipblasOperation_t transA,
                                           hipblasDiagType_t  diag,
                                           int                m,
                                           int                k,
                                           const float*       A,
                                           int                lda,
                                           hipblasStride      stride_a,
                                           float*             x,
                                           int                incx,
                                           hipblasStride      stride_x,
                                           int                batch_count)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDtbmvStridedBatched(hipblasHandle_t    handle,
                                           hipblasFillMode_t  uplo,
                                           hipblasOperation_t transA,
                                           hipblasDiagType_t  diag,
                                           int                m,
                                           int                k,
                                           const double*      A,
                                           int                lda,
                                           hipblasStride      stride_a,
                                           double*            x,
                                           int                incx,
                                           hipblasStride      stride_x,
                                           int                batch_count)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCtbmvStridedBatched(hipblasHandle_t       handle,
                                           hipblasFillMode_t     uplo,
                                           hipblasOperation_t    transA,
                                           hipblasDiagType_t     diag,
                                           int                   m,
                                           int                   k,
                                           const hipblasComplex* A,
                                           int                   lda,
                                           hipblasStride         stride_a,
                                           hipblasComplex*       x,
                                           int                   incx,
                                           hipblasStride         stride_x,
                                           int                   batch_count)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZtbmvStridedBatched(hipblasHandle_t             handle,
                                           hipblasFillMode_t           uplo,
                                           hipblasOperation_t          transA,
                                           hipblasDiagType_t           diag,
                                           int                         m,
                                           int                         k,
                                           const hipblasDoubleComplex* A,
                                           int                         lda,
                                           hipblasStride               stride_a,
                                           hipblasDoubleComplex*       x,
                                           int                         incx,
                                           hipblasStride               stride_x,
                                           int                         batch_count)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

// Level-2 : tbsv(supported datatypes : float , double , float complex and double complex )
hipblasStatus_t hipblasStbsv(hipblasHandle_t    handle,
                             hipblasFillMode_t  uplo,
                             hipblasOperation_t transA,
                             hipblasDiagType_t  diag,
                             int                n,
                             int                k,
                             const float*       A,
                             int                lda,
                             float*             x,
                             int                incx)
try
{
    if (handle == nullptr || A == nullptr || x == nullptr ||
        n <= 0 || k <= 0 || lda <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }

    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    onemklStbsv(sycl_queue, convert(uplo), convert(transA), convert(diag), n, k, A, lda, x, incx);
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDtbsv(hipblasHandle_t    handle,
                             hipblasFillMode_t  uplo,
                             hipblasOperation_t transA,
                             hipblasDiagType_t  diag,
                             int                n,
                             int                k,
                             const double*      A,
                             int                lda,
                             double*            x,
                             int                incx)
try
{
    if (handle == nullptr || A == nullptr || x == nullptr ||
        n <= 0 || k <= 0 || lda <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    onemklDtbsv(sycl_queue, convert(uplo), convert(transA), convert(diag), n, k, A, lda, x, incx);
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCtbsv(hipblasHandle_t       handle,
                             hipblasFillMode_t     uplo,
                             hipblasOperation_t    transA,
                             hipblasDiagType_t     diag,
                             int                   n,
                             int                   k,
                             const hipblasComplex* A,
                             int                   lda,
                             hipblasComplex*       x,
                             int                   incx)
try
{
    if (handle == nullptr || A == nullptr || x == nullptr ||
        n <= 0 || k <= 0 || lda <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    onemklCtbsv(sycl_queue, convert(uplo), convert(transA), convert(diag), n, k, (const float _Complex*)A, lda, (float _Complex*)x, incx);
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZtbsv(hipblasHandle_t             handle,
                             hipblasFillMode_t           uplo,
                             hipblasOperation_t          transA,
                             hipblasDiagType_t           diag,
                             int                         n,
                             int                         k,
                             const hipblasDoubleComplex* A,
                             int                         lda,
                             hipblasDoubleComplex*       x,
                             int                         incx)
try
{
    if (handle == nullptr || A == nullptr || x == nullptr ||
        n <= 0 || k <= 0 || lda <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    onemklZtbsv(sycl_queue, convert(uplo), convert(transA), convert(diag), n, k, (const double _Complex*)A, lda, (double _Complex*)x, incx);
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

// tbsv_batched
hipblasStatus_t hipblasStbsvBatched(hipblasHandle_t    handle,
                                    hipblasFillMode_t  uplo,
                                    hipblasOperation_t transA,
                                    hipblasDiagType_t  diag,
                                    int                n,
                                    int                k,
                                    const float* const A[],
                                    int                lda,
                                    float* const       x[],
                                    int                incx,
                                    int                batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDtbsvBatched(hipblasHandle_t     handle,
                                    hipblasFillMode_t   uplo,
                                    hipblasOperation_t  transA,
                                    hipblasDiagType_t   diag,
                                    int                 n,
                                    int                 k,
                                    const double* const A[],
                                    int                 lda,
                                    double* const       x[],
                                    int                 incx,
                                    int                 batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCtbsvBatched(hipblasHandle_t             handle,
                                    hipblasFillMode_t           uplo,
                                    hipblasOperation_t          transA,
                                    hipblasDiagType_t           diag,
                                    int                         n,
                                    int                         k,
                                    const hipblasComplex* const A[],
                                    int                         lda,
                                    hipblasComplex* const       x[],
                                    int                         incx,
                                    int                         batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZtbsvBatched(hipblasHandle_t                   handle,
                                    hipblasFillMode_t                 uplo,
                                    hipblasOperation_t                transA,
                                    hipblasDiagType_t                 diag,
                                    int                               n,
                                    int                               k,
                                    const hipblasDoubleComplex* const A[],
                                    int                               lda,
                                    hipblasDoubleComplex* const       x[],
                                    int                               incx,
                                    int                               batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

// tbsv_strided_batched
hipblasStatus_t hipblasStbsvStridedBatched(hipblasHandle_t    handle,
                                           hipblasFillMode_t  uplo,
                                           hipblasOperation_t transA,
                                           hipblasDiagType_t  diag,
                                           int                n,
                                           int                k,
                                           const float*       A,
                                           int                lda,
                                           hipblasStride      strideA,
                                           float*             x,
                                           int                incx,
                                           hipblasStride      stridex,
                                           int                batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDtbsvStridedBatched(hipblasHandle_t    handle,
                                           hipblasFillMode_t  uplo,
                                           hipblasOperation_t transA,
                                           hipblasDiagType_t  diag,
                                           int                n,
                                           int                k,
                                           const double*      A,
                                           int                lda,
                                           hipblasStride      strideA,
                                           double*            x,
                                           int                incx,
                                           hipblasStride      stridex,
                                           int                batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCtbsvStridedBatched(hipblasHandle_t       handle,
                                           hipblasFillMode_t     uplo,
                                           hipblasOperation_t    transA,
                                           hipblasDiagType_t     diag,
                                           int                   n,
                                           int                   k,
                                           const hipblasComplex* A,
                                           int                   lda,
                                           hipblasStride         strideA,
                                           hipblasComplex*       x,
                                           int                   incx,
                                           hipblasStride         stridex,
                                           int                   batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZtbsvStridedBatched(hipblasHandle_t             handle,
                                           hipblasFillMode_t           uplo,
                                           hipblasOperation_t          transA,
                                           hipblasDiagType_t           diag,
                                           int                         n,
                                           int                         k,
                                           const hipblasDoubleComplex* A,
                                           int                         lda,
                                           hipblasStride               strideA,
                                           hipblasDoubleComplex*       x,
                                           int                         incx,
                                           hipblasStride               stridex,
                                           int                         batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

// Level-2 : tpmv(supported datatypes : float , double , float complex and double complex )
hipblasStatus_t hipblasStpmv(hipblasHandle_t    handle,
                             hipblasFillMode_t  uplo,
                             hipblasOperation_t transA,
                             hipblasDiagType_t  diag,
                             int                m,
                             const float*       AP,
                             float*             x,
                             int                incx)
try
{
    if (handle == nullptr || AP == nullptr || x == nullptr ||
        m <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    onemklStpmv(sycl_queue, convert(uplo), convert(transA), convert(diag), m, AP, x, incx);
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDtpmv(hipblasHandle_t    handle,
                             hipblasFillMode_t  uplo,
                             hipblasOperation_t transA,
                             hipblasDiagType_t  diag,
                             int                m,
                             const double*      AP,
                             double*            x,
                             int                incx)
try
{
    if (handle == nullptr || AP == nullptr || x == nullptr ||
        m <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    onemklDtpmv(sycl_queue, convert(uplo), convert(transA), convert(diag), m, AP, x, incx);
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCtpmv(hipblasHandle_t       handle,
                             hipblasFillMode_t     uplo,
                             hipblasOperation_t    transA,
                             hipblasDiagType_t     diag,
                             int                   m,
                             const hipblasComplex* AP,
                             hipblasComplex*       x,
                             int                   incx)
try
{
    if (handle == nullptr || AP == nullptr || x == nullptr ||
        m <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    onemklCtpmv(sycl_queue, convert(uplo), convert(transA), convert(diag), m, 
                (const float _Complex*)AP, (float _Complex*)x, incx);
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZtpmv(hipblasHandle_t             handle,
                             hipblasFillMode_t           uplo,
                             hipblasOperation_t          transA,
                             hipblasDiagType_t           diag,
                             int                         m,
                             const hipblasDoubleComplex* AP,
                             hipblasDoubleComplex*       x,
                             int                         incx)
try
{
    if (handle == nullptr || AP == nullptr || x == nullptr ||
        m <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    onemklZtpmv(sycl_queue, convert(uplo), convert(transA), convert(diag), m, 
                (const double _Complex*)AP, (double _Complex*)x, incx);
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

// tpmv_batched
hipblasStatus_t hipblasStpmvBatched(hipblasHandle_t    handle,
                                    hipblasFillMode_t  uplo,
                                    hipblasOperation_t transA,
                                    hipblasDiagType_t  diag,
                                    int                m,
                                    const float* const AP[],
                                    float* const       x[],
                                    int                incx,
                                    int                batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDtpmvBatched(hipblasHandle_t     handle,
                                    hipblasFillMode_t   uplo,
                                    hipblasOperation_t  transA,
                                    hipblasDiagType_t   diag,
                                    int                 m,
                                    const double* const AP[],
                                    double* const       x[],
                                    int                 incx,
                                    int                 batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCtpmvBatched(hipblasHandle_t             handle,
                                    hipblasFillMode_t           uplo,
                                    hipblasOperation_t          transA,
                                    hipblasDiagType_t           diag,
                                    int                         m,
                                    const hipblasComplex* const AP[],
                                    hipblasComplex* const       x[],
                                    int                         incx,
                                    int                         batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZtpmvBatched(hipblasHandle_t                   handle,
                                    hipblasFillMode_t                 uplo,
                                    hipblasOperation_t                transA,
                                    hipblasDiagType_t                 diag,
                                    int                               m,
                                    const hipblasDoubleComplex* const AP[],
                                    hipblasDoubleComplex* const       x[],
                                    int                               incx,
                                    int                               batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

// tpmv_strided_batched
hipblasStatus_t hipblasStpmvStridedBatched(hipblasHandle_t    handle,
                                           hipblasFillMode_t  uplo,
                                           hipblasOperation_t transA,
                                           hipblasDiagType_t  diag,
                                           int                m,
                                           const float*       AP,
                                           hipblasStride      strideAP,
                                           float*             x,
                                           int                incx,
                                           hipblasStride      stridex,
                                           int                batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDtpmvStridedBatched(hipblasHandle_t    handle,
                                           hipblasFillMode_t  uplo,
                                           hipblasOperation_t transA,
                                           hipblasDiagType_t  diag,
                                           int                m,
                                           const double*      AP,
                                           hipblasStride      strideAP,
                                           double*            x,
                                           int                incx,
                                           hipblasStride      stridex,
                                           int                batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCtpmvStridedBatched(hipblasHandle_t       handle,
                                           hipblasFillMode_t     uplo,
                                           hipblasOperation_t    transA,
                                           hipblasDiagType_t     diag,
                                           int                   m,
                                           const hipblasComplex* AP,
                                           hipblasStride         strideAP,
                                           hipblasComplex*       x,
                                           int                   incx,
                                           hipblasStride         stridex,
                                           int                   batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZtpmvStridedBatched(hipblasHandle_t             handle,
                                           hipblasFillMode_t           uplo,
                                           hipblasOperation_t          transA,
                                           hipblasDiagType_t           diag,
                                           int                         m,
                                           const hipblasDoubleComplex* AP,
                                           hipblasStride               strideAP,
                                           hipblasDoubleComplex*       x,
                                           int                         incx,
                                           hipblasStride               stridex,
                                           int                         batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

// Level-2 : tpsv(supported datatypes : float , double , float complex and double complex )
hipblasStatus_t hipblasStpsv(hipblasHandle_t    handle,
                             hipblasFillMode_t  uplo,
                             hipblasOperation_t transA,
                             hipblasDiagType_t  diag,
                             int                m,
                             const float*       AP,
                             float*             x,
                             int                incx)
try
{
    if (handle == nullptr || AP == nullptr || x == nullptr ||
        m <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    onemklStpsv(sycl_queue, convert(uplo), convert(transA), convert(diag), m, AP, x, incx);
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDtpsv(hipblasHandle_t    handle,
                             hipblasFillMode_t  uplo,
                             hipblasOperation_t transA,
                             hipblasDiagType_t  diag,
                             int                m,
                             const double*      AP,
                             double*            x,
                             int                incx)
try
{
    if (handle == nullptr || AP == nullptr || x == nullptr ||
        m <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    onemklDtpsv(sycl_queue, convert(uplo), convert(transA), convert(diag), m, AP, x, incx);
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCtpsv(hipblasHandle_t       handle,
                             hipblasFillMode_t     uplo,
                             hipblasOperation_t    transA,
                             hipblasDiagType_t     diag,
                             int                   m,
                             const hipblasComplex* AP,
                             hipblasComplex*       x,
                             int                   incx)
try
{
    if (handle == nullptr || AP == nullptr || x == nullptr ||
        m <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    onemklCtpsv(sycl_queue, convert(uplo), convert(transA), convert(diag), m,
                (const float _Complex*) AP, (float _Complex*)x, incx);
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZtpsv(hipblasHandle_t             handle,
                             hipblasFillMode_t           uplo,
                             hipblasOperation_t          transA,
                             hipblasDiagType_t           diag,
                             int                         m,
                             const hipblasDoubleComplex* AP,
                             hipblasDoubleComplex*       x,
                             int                         incx)
try
{
    if (handle == nullptr || AP == nullptr || x == nullptr ||
        m <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    onemklZtpsv(sycl_queue, convert(uplo), convert(transA), convert(diag), m,
                (const double _Complex*) AP, (double _Complex*)x, incx);
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

// tpsv_batched
hipblasStatus_t hipblasStpsvBatched(hipblasHandle_t    handle,
                                    hipblasFillMode_t  uplo,
                                    hipblasOperation_t transA,
                                    hipblasDiagType_t  diag,
                                    int                m,
                                    const float* const AP[],
                                    float* const       x[],
                                    int                incx,
                                    int                batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDtpsvBatched(hipblasHandle_t     handle,
                                    hipblasFillMode_t   uplo,
                                    hipblasOperation_t  transA,
                                    hipblasDiagType_t   diag,
                                    int                 m,
                                    const double* const AP[],
                                    double* const       x[],
                                    int                 incx,
                                    int                 batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCtpsvBatched(hipblasHandle_t             handle,
                                    hipblasFillMode_t           uplo,
                                    hipblasOperation_t          transA,
                                    hipblasDiagType_t           diag,
                                    int                         m,
                                    const hipblasComplex* const AP[],
                                    hipblasComplex* const       x[],
                                    int                         incx,
                                    int                         batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZtpsvBatched(hipblasHandle_t                   handle,
                                    hipblasFillMode_t                 uplo,
                                    hipblasOperation_t                transA,
                                    hipblasDiagType_t                 diag,
                                    int                               m,
                                    const hipblasDoubleComplex* const AP[],
                                    hipblasDoubleComplex* const       x[],
                                    int                               incx,
                                    int                               batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

// tpsv_strided_batched
hipblasStatus_t hipblasStpsvStridedBatched(hipblasHandle_t    handle,
                                           hipblasFillMode_t  uplo,
                                           hipblasOperation_t transA,
                                           hipblasDiagType_t  diag,
                                           int                m,
                                           const float*       AP,
                                           hipblasStride      strideAP,
                                           float*             x,
                                           int                incx,
                                           hipblasStride      stridex,
                                           int                batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDtpsvStridedBatched(hipblasHandle_t    handle,
                                           hipblasFillMode_t  uplo,
                                           hipblasOperation_t transA,
                                           hipblasDiagType_t  diag,
                                           int                m,
                                           const double*      AP,
                                           hipblasStride      strideAP,
                                           double*            x,
                                           int                incx,
                                           hipblasStride      stridex,
                                           int                batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCtpsvStridedBatched(hipblasHandle_t       handle,
                                           hipblasFillMode_t     uplo,
                                           hipblasOperation_t    transA,
                                           hipblasDiagType_t     diag,
                                           int                   m,
                                           const hipblasComplex* AP,
                                           hipblasStride         strideAP,
                                           hipblasComplex*       x,
                                           int                   incx,
                                           hipblasStride         stridex,
                                           int                   batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZtpsvStridedBatched(hipblasHandle_t             handle,
                                           hipblasFillMode_t           uplo,
                                           hipblasOperation_t          transA,
                                           hipblasDiagType_t           diag,
                                           int                         m,
                                           const hipblasDoubleComplex* AP,
                                           hipblasStride               strideAP,
                                           hipblasDoubleComplex*       x,
                                           int                         incx,
                                           hipblasStride               stridex,
                                           int                         batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

// Level-2 : trmv(supported datatypes : float , double , float complex and double complex )
hipblasStatus_t hipblasStrmv(hipblasHandle_t    handle,
                             hipblasFillMode_t  uplo,
                             hipblasOperation_t transA,
                             hipblasDiagType_t  diag,
                             int                m,
                             const float*       A,
                             int                lda,
                             float*             x,
                             int                incx)
try
{
    if (handle == nullptr || A == nullptr || x == nullptr ||
        m <= 0 || lda <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    onemklStrmv(sycl_queue, convert(uplo), convert(transA), convert(diag), m, A, lda, x, incx);
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDtrmv(hipblasHandle_t    handle,
                             hipblasFillMode_t  uplo,
                             hipblasOperation_t transA,
                             hipblasDiagType_t  diag,
                             int                m,
                             const double*      A,
                             int                lda,
                             double*            x,
                             int                incx)
try
{
    if (handle == nullptr || A == nullptr || x == nullptr ||
        m <= 0 || lda <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    onemklDtrmv(sycl_queue, convert(uplo), convert(transA), convert(diag), m, A, lda, x, incx);
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCtrmv(hipblasHandle_t       handle,
                             hipblasFillMode_t     uplo,
                             hipblasOperation_t    transA,
                             hipblasDiagType_t     diag,
                             int                   m,
                             const hipblasComplex* A,
                             int                   lda,
                             hipblasComplex*       x,
                             int                   incx)
try
{
    if (handle == nullptr || A == nullptr || x == nullptr ||
        m <= 0 || lda <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    onemklCtrmv(sycl_queue, convert(uplo), convert(transA), convert(diag), m,
                (const float _Complex*)A, lda, (float _Complex*)x, incx);
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZtrmv(hipblasHandle_t             handle,
                             hipblasFillMode_t           uplo,
                             hipblasOperation_t          transA,
                             hipblasDiagType_t           diag,
                             int                         m,
                             const hipblasDoubleComplex* A,
                             int                         lda,
                             hipblasDoubleComplex*       x,
                             int                         incx)
try
{
    if (handle == nullptr || A == nullptr || x == nullptr ||
        m <= 0 || lda <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    onemklZtrmv(sycl_queue, convert(uplo), convert(transA), convert(diag), m,
                (const double _Complex*)A, lda, (double _Complex*)x, incx);
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

// trmv_batched
hipblasStatus_t hipblasStrmvBatched(hipblasHandle_t    handle,
                                    hipblasFillMode_t  uplo,
                                    hipblasOperation_t transA,
                                    hipblasDiagType_t  diag,
                                    int                m,
                                    const float* const A[],
                                    int                lda,
                                    float* const       x[],
                                    int                incx,
                                    int                batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDtrmvBatched(hipblasHandle_t     handle,
                                    hipblasFillMode_t   uplo,
                                    hipblasOperation_t  transA,
                                    hipblasDiagType_t   diag,
                                    int                 m,
                                    const double* const A[],
                                    int                 lda,
                                    double* const       x[],
                                    int                 incx,
                                    int                 batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCtrmvBatched(hipblasHandle_t             handle,
                                    hipblasFillMode_t           uplo,
                                    hipblasOperation_t          transA,
                                    hipblasDiagType_t           diag,
                                    int                         m,
                                    const hipblasComplex* const A[],
                                    int                         lda,
                                    hipblasComplex* const       x[],
                                    int                         incx,
                                    int                         batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZtrmvBatched(hipblasHandle_t                   handle,
                                    hipblasFillMode_t                 uplo,
                                    hipblasOperation_t                transA,
                                    hipblasDiagType_t                 diag,
                                    int                               m,
                                    const hipblasDoubleComplex* const A[],
                                    int                               lda,
                                    hipblasDoubleComplex* const       x[],
                                    int                               incx,
                                    int                               batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

// trmv_strided_batched
hipblasStatus_t hipblasStrmvStridedBatched(hipblasHandle_t    handle,
                                           hipblasFillMode_t  uplo,
                                           hipblasOperation_t transA,
                                           hipblasDiagType_t  diag,
                                           int                m,
                                           const float*       A,
                                           int                lda,
                                           hipblasStride      stride_a,
                                           float*             x,
                                           int                incx,
                                           hipblasStride      stridex,
                                           int                batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDtrmvStridedBatched(hipblasHandle_t    handle,
                                           hipblasFillMode_t  uplo,
                                           hipblasOperation_t transA,
                                           hipblasDiagType_t  diag,
                                           int                m,
                                           const double*      A,
                                           int                lda,
                                           hipblasStride      stride_a,
                                           double*            x,
                                           int                incx,
                                           hipblasStride      stridex,
                                           int                batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCtrmvStridedBatched(hipblasHandle_t       handle,
                                           hipblasFillMode_t     uplo,
                                           hipblasOperation_t    transA,
                                           hipblasDiagType_t     diag,
                                           int                   m,
                                           const hipblasComplex* A,
                                           int                   lda,
                                           hipblasStride         stride_a,
                                           hipblasComplex*       x,
                                           int                   incx,
                                           hipblasStride         stridex,
                                           int                   batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZtrmvStridedBatched(hipblasHandle_t             handle,
                                           hipblasFillMode_t           uplo,
                                           hipblasOperation_t          transA,
                                           hipblasDiagType_t           diag,
                                           int                         m,
                                           const hipblasDoubleComplex* A,
                                           int                         lda,
                                           hipblasStride               stride_a,
                                           hipblasDoubleComplex*       x,
                                           int                         incx,
                                           hipblasStride               stridex,
                                           int                         batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

// Level-2 : trsv(supported datatypes : float , double , float complex and double complex )
hipblasStatus_t hipblasStrsv(hipblasHandle_t    handle,
                             hipblasFillMode_t  uplo,
                             hipblasOperation_t transA,
                             hipblasDiagType_t  diag,
                             int                m,
                             const float*       A,
                             int                lda,
                             float*             x,
                             int                incx)
try
{
    if (handle == nullptr || A == nullptr || x == nullptr ||
        m <= 0 ||lda <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    onemklStrsv(sycl_queue, convert(uplo), convert(transA), convert(diag), m, A, lda, x, incx);
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDtrsv(hipblasHandle_t    handle,
                             hipblasFillMode_t  uplo,
                             hipblasOperation_t transA,
                             hipblasDiagType_t  diag,
                             int                m,
                             const double*      A,
                             int                lda,
                             double*            x,
                             int                incx)
try
{
    if (handle == nullptr || A == nullptr || x == nullptr ||
        m <= 0 ||lda <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    onemklDtrsv(sycl_queue, convert(uplo), convert(transA), convert(diag), m, A, lda, x, incx);
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCtrsv(hipblasHandle_t       handle,
                             hipblasFillMode_t     uplo,
                             hipblasOperation_t    transA,
                             hipblasDiagType_t     diag,
                             int                   m,
                             const hipblasComplex* A,
                             int                   lda,
                             hipblasComplex*       x,
                             int                   incx)
try
{
    if (handle == nullptr || A == nullptr || x == nullptr ||
        m <= 0 ||lda <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    onemklCtrsv(sycl_queue, convert(uplo), convert(transA), convert(diag), m,
                (const float _Complex*)A, lda, (float _Complex*)x, incx);
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZtrsv(hipblasHandle_t             handle,
                             hipblasFillMode_t           uplo,
                             hipblasOperation_t          transA,
                             hipblasDiagType_t           diag,
                             int                         m,
                             const hipblasDoubleComplex* A,
                             int                         lda,
                             hipblasDoubleComplex*       x,
                             int                         incx)
try
{
    if (handle == nullptr || A == nullptr || x == nullptr ||
        m <= 0 ||lda <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    onemklZtrsv(sycl_queue, convert(uplo), convert(transA), convert(diag), m,
                (const double _Complex*)A, lda, (double _Complex*)x, incx);
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

// trsv_batched
hipblasStatus_t hipblasStrsvBatched(hipblasHandle_t    handle,
                                    hipblasFillMode_t  uplo,
                                    hipblasOperation_t transA,
                                    hipblasDiagType_t  diag,
                                    int                m,
                                    const float* const A[],
                                    int                lda,
                                    float* const       x[],
                                    int                incx,
                                    int                batch_count)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDtrsvBatched(hipblasHandle_t     handle,
                                    hipblasFillMode_t   uplo,
                                    hipblasOperation_t  transA,
                                    hipblasDiagType_t   diag,
                                    int                 m,
                                    const double* const A[],
                                    int                 lda,
                                    double* const       x[],
                                    int                 incx,
                                    int                 batch_count)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCtrsvBatched(hipblasHandle_t             handle,
                                    hipblasFillMode_t           uplo,
                                    hipblasOperation_t          transA,
                                    hipblasDiagType_t           diag,
                                    int                         m,
                                    const hipblasComplex* const A[],
                                    int                         lda,
                                    hipblasComplex* const       x[],
                                    int                         incx,
                                    int                         batch_count)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZtrsvBatched(hipblasHandle_t                   handle,
                                    hipblasFillMode_t                 uplo,
                                    hipblasOperation_t                transA,
                                    hipblasDiagType_t                 diag,
                                    int                               m,
                                    const hipblasDoubleComplex* const A[],
                                    int                               lda,
                                    hipblasDoubleComplex* const       x[],
                                    int                               incx,
                                    int                               batch_count)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

// trsv_strided_batched
hipblasStatus_t hipblasStrsvStridedBatched(hipblasHandle_t    handle,
                                           hipblasFillMode_t  uplo,
                                           hipblasOperation_t transA,
                                           hipblasDiagType_t  diag,
                                           int                m,
                                           const float*       A,
                                           int                lda,
                                           hipblasStride      strideA,
                                           float*             x,
                                           int                incx,
                                           hipblasStride      stridex,
                                           int                batch_count)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDtrsvStridedBatched(hipblasHandle_t    handle,
                                           hipblasFillMode_t  uplo,
                                           hipblasOperation_t transA,
                                           hipblasDiagType_t  diag,
                                           int                m,
                                           const double*      A,
                                           int                lda,
                                           hipblasStride      strideA,
                                           double*            x,
                                           int                incx,
                                           hipblasStride      stridex,
                                           int                batch_count)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCtrsvStridedBatched(hipblasHandle_t       handle,
                                           hipblasFillMode_t     uplo,
                                           hipblasOperation_t    transA,
                                           hipblasDiagType_t     diag,
                                           int                   m,
                                           const hipblasComplex* A,
                                           int                   lda,
                                           hipblasStride         strideA,
                                           hipblasComplex*       x,
                                           int                   incx,
                                           hipblasStride         stridex,
                                           int                   batch_count)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZtrsvStridedBatched(hipblasHandle_t             handle,
                                           hipblasFillMode_t           uplo,
                                           hipblasOperation_t          transA,
                                           hipblasDiagType_t           diag,
                                           int                         m,
                                           const hipblasDoubleComplex* A,
                                           int                         lda,
                                           hipblasStride               strideA,
                                           hipblasDoubleComplex*       x,
                                           int                         incx,
                                           hipblasStride               stridex,
                                           int                         batch_count)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

//------------------------------------------------------------------------------------------------------------

// Level-3 : herk(supported datatypes : float complex and double complex )
hipblasStatus_t hipblasCherk(hipblasHandle_t       handle,
                             hipblasFillMode_t     uplo,
                             hipblasOperation_t    transA,
                             int                   n,
                             int                   k,
                             const float*          alpha,
                             const hipblasComplex* A,
                             int                   lda,
                             const float*          beta,
                             hipblasComplex*       C,
                             int                   ldc)
try
{
    if (handle == nullptr || alpha == nullptr || A == nullptr || C == nullptr || beta == nullptr ||
        k <= 0 || n <= 0 || lda <= 0 || ldc <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    // As per spec alpha can be device/host memory. In case of device memory *alpha will crash
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    auto is_dev_ptr = (queryCurrentPtrMode(handle) == HIPBLAS_POINTER_MODE_DEVICE);

    float h_alpha, h_beta;
    if (is_dev_ptr) {
        hipMemcpy(&h_alpha, alpha, sizeof(float), hipMemcpyDefault);
        hipMemcpy(&h_beta, beta, sizeof(float), hipMemcpyDefault);
    } else {
        h_alpha = *((float*)alpha);
        h_beta = *((float*)beta);
    }

    onemklCherk(sycl_queue, convert(uplo), convert(transA), n, k,
                h_alpha, (const float _Complex*)A, lda, h_beta, (float _Complex*)C, ldc);
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZherk(hipblasHandle_t             handle,
                             hipblasFillMode_t           uplo,
                             hipblasOperation_t          transA,
                             int                         n,
                             int                         k,
                             const double*               alpha,
                             const hipblasDoubleComplex* A,
                             int                         lda,
                             const double*               beta,
                             hipblasDoubleComplex*       C,
                             int                         ldc)
try
{
    if (handle == nullptr || alpha == nullptr || A == nullptr || C == nullptr || beta == nullptr ||
        k <= 0 || n <= 0 || lda <= 0 || ldc <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    // As per spec alpha can be device/host memory. In case of device memory *alpha will crash
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    auto is_dev_ptr = (queryCurrentPtrMode(handle) == HIPBLAS_POINTER_MODE_DEVICE);

    double h_alpha, h_beta;
    if (is_dev_ptr) {
        hipMemcpy(&h_alpha, alpha, sizeof(double), hipMemcpyDefault);
        hipMemcpy(&h_beta, beta, sizeof(double), hipMemcpyDefault);
    } else {
        h_alpha = *((double*)alpha);
        h_beta = *((double*)beta);
    }
    onemklZherk(sycl_queue, convert(uplo), convert(transA), n, k,
                h_alpha, (const double _Complex*)A, lda, h_beta, (double _Complex*)C, ldc);
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

// herk_batched
hipblasStatus_t hipblasCherkBatched(hipblasHandle_t             handle,
                                    hipblasFillMode_t           uplo,
                                    hipblasOperation_t          transA,
                                    int                         n,
                                    int                         k,
                                    const float*                alpha,
                                    const hipblasComplex* const A[],
                                    int                         lda,
                                    const float*                beta,
                                    hipblasComplex* const       C[],
                                    int                         ldc,
                                    int                         batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZherkBatched(hipblasHandle_t                   handle,
                                    hipblasFillMode_t                 uplo,
                                    hipblasOperation_t                transA,
                                    int                               n,
                                    int                               k,
                                    const double*                     alpha,
                                    const hipblasDoubleComplex* const A[],
                                    int                               lda,
                                    const double*                     beta,
                                    hipblasDoubleComplex* const       C[],
                                    int                               ldc,
                                    int                               batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

// herk_strided_batched
hipblasStatus_t hipblasCherkStridedBatched(hipblasHandle_t       handle,
                                           hipblasFillMode_t     uplo,
                                           hipblasOperation_t    transA,
                                           int                   n,
                                           int                   k,
                                           const float*          alpha,
                                           const hipblasComplex* A,
                                           int                   lda,
                                           hipblasStride         strideA,
                                           const float*          beta,
                                           hipblasComplex*       C,
                                           int                   ldc,
                                           hipblasStride         strideC,
                                           int                   batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZherkStridedBatched(hipblasHandle_t             handle,
                                           hipblasFillMode_t           uplo,
                                           hipblasOperation_t          transA,
                                           int                         n,
                                           int                         k,
                                           const double*               alpha,
                                           const hipblasDoubleComplex* A,
                                           int                         lda,
                                           hipblasStride               strideA,
                                           const double*               beta,
                                           hipblasDoubleComplex*       C,
                                           int                         ldc,
                                           hipblasStride               strideC,
                                           int                         batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

// Level-2 : herkx(supported datatypes : float complex , double complex )
hipblasStatus_t hipblasCherkx(hipblasHandle_t       handle,
                              hipblasFillMode_t     uplo,
                              hipblasOperation_t    transA,
                              int                   n,
                              int                   k,
                              const hipblasComplex* alpha,
                              const hipblasComplex* A,
                              int                   lda,
                              const hipblasComplex* B,
                              int                   ldb,
                              const float*          beta,
                              hipblasComplex*       C,
                              int                   ldc)
try
{
return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZherkx(hipblasHandle_t             handle,
                              hipblasFillMode_t           uplo,
                              hipblasOperation_t          transA,
                              int                         n,
                              int                         k,
                              const hipblasDoubleComplex* alpha,
                              const hipblasDoubleComplex* A,
                              int                         lda,
                              const hipblasDoubleComplex* B,
                              int                         ldb,
                              const double*               beta,
                              hipblasDoubleComplex*       C,
                              int                         ldc)
try
{
return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

// herkx_batched
hipblasStatus_t hipblasCherkxBatched(hipblasHandle_t             handle,
                                     hipblasFillMode_t           uplo,
                                     hipblasOperation_t          transA,
                                     int                         n,
                                     int                         k,
                                     const hipblasComplex*       alpha,
                                     const hipblasComplex* const A[],
                                     int                         lda,
                                     const hipblasComplex* const B[],
                                     int                         ldb,
                                     const float*                beta,
                                     hipblasComplex* const       C[],
                                     int                         ldc,
                                     int                         batchCount)
try
{
return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZherkxBatched(hipblasHandle_t                   handle,
                                     hipblasFillMode_t                 uplo,
                                     hipblasOperation_t                transA,
                                     int                               n,
                                     int                               k,
                                     const hipblasDoubleComplex*       alpha,
                                     const hipblasDoubleComplex* const A[],
                                     int                               lda,
                                     const hipblasDoubleComplex* const B[],
                                     int                               ldb,
                                     const double*                     beta,
                                     hipblasDoubleComplex* const       C[],
                                     int                               ldc,
                                     int                               batchCount)
try
{
return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

// herkx_strided_batched
hipblasStatus_t hipblasCherkxStridedBatched(hipblasHandle_t       handle,
                                            hipblasFillMode_t     uplo,
                                            hipblasOperation_t    transA,
                                            int                   n,
                                            int                   k,
                                            const hipblasComplex* alpha,
                                            const hipblasComplex* A,
                                            int                   lda,
                                            hipblasStride         strideA,
                                            const hipblasComplex* B,
                                            int                   ldb,
                                            hipblasStride         strideB,
                                            const float*          beta,
                                            hipblasComplex*       C,
                                            int                   ldc,
                                            hipblasStride         strideC,
                                            int                   batchCount)
try
{
return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZherkxStridedBatched(hipblasHandle_t             handle,
                                            hipblasFillMode_t           uplo,
                                            hipblasOperation_t          transA,
                                            int                         n,
                                            int                         k,
                                            const hipblasDoubleComplex* alpha,
                                            const hipblasDoubleComplex* A,
                                            int                         lda,
                                            hipblasStride               strideA,
                                            const hipblasDoubleComplex* B,
                                            int                         ldb,
                                            hipblasStride               strideB,
                                            const double*               beta,
                                            hipblasDoubleComplex*       C,
                                            int                         ldc,
                                            hipblasStride               strideC,
                                            int                         batchCount)
try
{
return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

// Level-3 : her2k(supported datatypes : float complex and double complex )
hipblasStatus_t hipblasCher2k(hipblasHandle_t       handle,
                              hipblasFillMode_t     uplo,
                              hipblasOperation_t    transA,
                              int                   n,
                              int                   k,
                              const hipblasComplex* alpha,
                              const hipblasComplex* A,
                              int                   lda,
                              const hipblasComplex* B,
                              int                   ldb,
                              const float*          beta,
                              hipblasComplex*       C,
                              int                   ldc)
try
{
    if (handle == nullptr || alpha == nullptr || A == nullptr || B == nullptr || C == nullptr || beta == nullptr ||
        k <= 0 || n <= 0 || lda <= 0 || ldb <= 0 || ldc <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    // As per spec alpha can be device/host memory. In case of device memory *alpha will crash
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    auto is_dev_ptr = (queryCurrentPtrMode(handle) == HIPBLAS_POINTER_MODE_DEVICE);

    float _Complex h_alpha;
    float h_beta;
    if (is_dev_ptr) {
        hipMemcpy(&h_alpha, alpha, sizeof(float _Complex), hipMemcpyDefault);
        hipMemcpy(&h_beta, beta, sizeof(float), hipMemcpyDefault);
    } else {
        h_alpha = *((float _Complex*)alpha);
        h_beta = *((float*)beta);
    }

    onemklCher2k(sycl_queue, convert(uplo), convert(transA), n, k, h_alpha,
                (const float _Complex*)A, lda, (const float _Complex*)B, ldb, h_beta,
                (float _Complex*)C, ldc);
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZher2k(hipblasHandle_t             handle,
                              hipblasFillMode_t           uplo,
                              hipblasOperation_t          transA,
                              int                         n,
                              int                         k,
                              const hipblasDoubleComplex* alpha,
                              const hipblasDoubleComplex* A,
                              int                         lda,
                              const hipblasDoubleComplex* B,
                              int                         ldb,
                              const double*               beta,
                              hipblasDoubleComplex*       C,
                              int                         ldc)
try
{
    if (handle == nullptr || alpha == nullptr || A == nullptr || B == nullptr || C == nullptr || beta == nullptr ||
        k <= 0 || n <= 0 || lda <= 0 || ldb <= 0 || ldc <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    // As per spec alpha can be device/host memory. In case of device memory *alpha will crash
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    auto is_dev_ptr = (queryCurrentPtrMode(handle) == HIPBLAS_POINTER_MODE_DEVICE);

    double _Complex h_alpha;
    double h_beta;
    if (is_dev_ptr) {
        hipMemcpy(&h_alpha, alpha, sizeof(double _Complex), hipMemcpyDefault);
        hipMemcpy(&h_beta, beta, sizeof(double), hipMemcpyDefault);
    } else {
        h_alpha = *((double _Complex*)alpha);
        h_beta = *((double*)beta);
    }

    onemklZher2k(sycl_queue, convert(uplo), convert(transA), n, k, h_alpha,
                (const double _Complex*)A, lda, (const double _Complex*)B, ldb, h_beta,
                (double _Complex*)C, ldc);
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

// her2k_batched
hipblasStatus_t hipblasCher2kBatched(hipblasHandle_t             handle,
                                     hipblasFillMode_t           uplo,
                                     hipblasOperation_t          transA,
                                     int                         n,
                                     int                         k,
                                     const hipblasComplex*       alpha,
                                     const hipblasComplex* const A[],
                                     int                         lda,
                                     const hipblasComplex* const B[],
                                     int                         ldb,
                                     const float*                beta,
                                     hipblasComplex* const       C[],
                                     int                         ldc,
                                     int                         batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZher2kBatched(hipblasHandle_t                   handle,
                                     hipblasFillMode_t                 uplo,
                                     hipblasOperation_t                transA,
                                     int                               n,
                                     int                               k,
                                     const hipblasDoubleComplex*       alpha,
                                     const hipblasDoubleComplex* const A[],
                                     int                               lda,
                                     const hipblasDoubleComplex* const B[],
                                     int                               ldb,
                                     const double*                     beta,
                                     hipblasDoubleComplex* const       C[],
                                     int                               ldc,
                                     int                               batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

// her2k_strided_batched
hipblasStatus_t hipblasCher2kStridedBatched(hipblasHandle_t       handle,
                                            hipblasFillMode_t     uplo,
                                            hipblasOperation_t    transA,
                                            int                   n,
                                            int                   k,
                                            const hipblasComplex* alpha,
                                            const hipblasComplex* A,
                                            int                   lda,
                                            hipblasStride         strideA,
                                            const hipblasComplex* B,
                                            int                   ldb,
                                            hipblasStride         strideB,
                                            const float*          beta,
                                            hipblasComplex*       C,
                                            int                   ldc,
                                            hipblasStride         strideC,
                                            int                   batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZher2kStridedBatched(hipblasHandle_t             handle,
                                            hipblasFillMode_t           uplo,
                                            hipblasOperation_t          transA,
                                            int                         n,
                                            int                         k,
                                            const hipblasDoubleComplex* alpha,
                                            const hipblasDoubleComplex* A,
                                            int                         lda,
                                            hipblasStride               strideA,
                                            const hipblasDoubleComplex* B,
                                            int                         ldb,
                                            hipblasStride               strideB,
                                            const double*               beta,
                                            hipblasDoubleComplex*       C,
                                            int                         ldc,
                                            hipblasStride               strideC,
                                            int                         batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}
// Level-3 : symm(supported datatypes : float , double , float complex and double complex )
hipblasStatus_t hipblasSsymm(hipblasHandle_t   handle,
                             hipblasSideMode_t side,
                             hipblasFillMode_t uplo,
                             int               m,
                             int               n,
                             const float*      alpha,
                             const float*      A,
                             int               lda,
                             const float*      B,
                             int               ldb,
                             const float*      beta,
                             float*            C,
                             int               ldc)
try
{
    if (handle == nullptr || alpha == nullptr || A == nullptr || B == nullptr || C == nullptr || beta == nullptr ||
        m <= 0 || n <= 0 || lda <= 0 || ldb <= 0 || ldc <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    // As per spec alpha can be device/host memory. In case of device memory *alpha will crash
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    auto is_dev_ptr = (queryCurrentPtrMode(handle) == HIPBLAS_POINTER_MODE_DEVICE);

    float h_alpha, h_beta;
    if (is_dev_ptr) {
        hipMemcpy(&h_alpha, alpha, sizeof(float), hipMemcpyDefault);
        hipMemcpy(&h_beta, beta, sizeof(float), hipMemcpyDefault);
    } else {
        h_alpha = *((float*)alpha);
        h_beta = *((float*)beta);
    }

    onemklSsymm(sycl_queue, convert(side), convert(uplo), m, n, h_alpha, A, lda, B, ldb, h_beta, C, ldc);
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDsymm(hipblasHandle_t   handle,
                             hipblasSideMode_t side,
                             hipblasFillMode_t uplo,
                             int               m,
                             int               n,
                             const double*     alpha,
                             const double*     A,
                             int               lda,
                             const double*     B,
                             int               ldb,
                             const double*     beta,
                             double*           C,
                             int               ldc)
try
{
    if (handle == nullptr || alpha == nullptr || A == nullptr || B == nullptr || C == nullptr || beta == nullptr ||
        m <= 0 || n <= 0 || lda <= 0 || ldb <= 0 || ldc <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    // As per spec alpha can be device/host memory. In case of device memory *alpha will crash
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    auto is_dev_ptr = (queryCurrentPtrMode(handle) == HIPBLAS_POINTER_MODE_DEVICE);

    double h_alpha, h_beta;
    if (is_dev_ptr) {
        hipMemcpy(&h_alpha, alpha, sizeof(double), hipMemcpyDefault);
        hipMemcpy(&h_beta, beta, sizeof(double), hipMemcpyDefault);
    } else {
        h_alpha = *((double*)alpha);
        h_beta = *((double*)beta);
    }

    onemklDsymm(sycl_queue, convert(side), convert(uplo), m, n, h_alpha, A, lda, B, ldb, h_beta, C, ldc);
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCsymm(hipblasHandle_t       handle,
                             hipblasSideMode_t     side,
                             hipblasFillMode_t     uplo,
                             int                   m,
                             int                   n,
                             const hipblasComplex* alpha,
                             const hipblasComplex* A,
                             int                   lda,
                             const hipblasComplex* B,
                             int                   ldb,
                             const hipblasComplex* beta,
                             hipblasComplex*       C,
                             int                   ldc)
try
{
    if (handle == nullptr || alpha == nullptr || A == nullptr || B == nullptr || C == nullptr || beta == nullptr ||
        m <= 0 || n <= 0 || lda <= 0 || ldb <= 0 || ldc <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    // As per spec alpha can be device/host memory. In case of device memory *alpha will crash
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    auto is_dev_ptr = (queryCurrentPtrMode(handle) == HIPBLAS_POINTER_MODE_DEVICE);

    float _Complex h_alpha, h_beta;
    if (is_dev_ptr) {
        hipMemcpy(&h_alpha, alpha, sizeof(float _Complex), hipMemcpyDefault);
        hipMemcpy(&h_beta, beta, sizeof(float _Complex), hipMemcpyDefault);
    } else {
        h_alpha = *((float _Complex*)alpha);
        h_beta = *((float _Complex*)beta);
    }

    onemklCsymm(sycl_queue, convert(side), convert(uplo), m, n, h_alpha,
                (const float _Complex*)A, lda,(const float _Complex*)B, ldb,
                h_beta, (float _Complex*)C, ldc);
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZsymm(hipblasHandle_t             handle,
                             hipblasSideMode_t           side,
                             hipblasFillMode_t           uplo,
                             int                         m,
                             int                         n,
                             const hipblasDoubleComplex* alpha,
                             const hipblasDoubleComplex* A,
                             int                         lda,
                             const hipblasDoubleComplex* B,
                             int                         ldb,
                             const hipblasDoubleComplex* beta,
                             hipblasDoubleComplex*       C,
                             int                         ldc)
try
{
    if (handle == nullptr || alpha == nullptr || A == nullptr || B == nullptr || C == nullptr || beta == nullptr ||
        m <= 0 || n <= 0 || lda <= 0 || ldb <= 0 || ldc <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    // As per spec alpha can be device/host memory. In case of device memory *alpha will crash
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    auto is_dev_ptr = (queryCurrentPtrMode(handle) == HIPBLAS_POINTER_MODE_DEVICE);

    double _Complex h_alpha, h_beta;
    if (is_dev_ptr) {
        hipMemcpy(&h_alpha, alpha, sizeof(double _Complex), hipMemcpyDefault);
        hipMemcpy(&h_beta, beta, sizeof(double _Complex), hipMemcpyDefault);
    } else {
        h_alpha = *((double _Complex*)alpha);
        h_beta = *((double _Complex*)beta);
    }

    onemklZsymm(sycl_queue, convert(side), convert(uplo), m, n, h_alpha,
                (const double _Complex*)A, lda,(const double _Complex*)B, ldb,
                h_beta, (double _Complex*)C, ldc);
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

// symm_batched
hipblasStatus_t hipblasSsymmBatched(hipblasHandle_t    handle,
                                    hipblasSideMode_t  side,
                                    hipblasFillMode_t  uplo,
                                    int                m,
                                    int                n,
                                    const float*       alpha,
                                    const float* const A[],
                                    int                lda,
                                    const float* const B[],
                                    int                ldb,
                                    const float*       beta,
                                    float* const       C[],
                                    int                ldc,
                                    int                batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDsymmBatched(hipblasHandle_t     handle,
                                    hipblasSideMode_t   side,
                                    hipblasFillMode_t   uplo,
                                    int                 m,
                                    int                 n,
                                    const double*       alpha,
                                    const double* const A[],
                                    int                 lda,
                                    const double* const B[],
                                    int                 ldb,
                                    const double*       beta,
                                    double* const       C[],
                                    int                 ldc,
                                    int                 batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCsymmBatched(hipblasHandle_t             handle,
                                    hipblasSideMode_t           side,
                                    hipblasFillMode_t           uplo,
                                    int                         m,
                                    int                         n,
                                    const hipblasComplex*       alpha,
                                    const hipblasComplex* const A[],
                                    int                         lda,
                                    const hipblasComplex* const B[],
                                    int                         ldb,
                                    const hipblasComplex*       beta,
                                    hipblasComplex* const       C[],
                                    int                         ldc,
                                    int                         batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZsymmBatched(hipblasHandle_t                   handle,
                                    hipblasSideMode_t                 side,
                                    hipblasFillMode_t                 uplo,
                                    int                               m,
                                    int                               n,
                                    const hipblasDoubleComplex*       alpha,
                                    const hipblasDoubleComplex* const A[],
                                    int                               lda,
                                    const hipblasDoubleComplex* const B[],
                                    int                               ldb,
                                    const hipblasDoubleComplex*       beta,
                                    hipblasDoubleComplex* const       C[],
                                    int                               ldc,
                                    int                               batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

// symm_strided_batched
hipblasStatus_t hipblasSsymmStridedBatched(hipblasHandle_t   handle,
                                           hipblasSideMode_t side,
                                           hipblasFillMode_t uplo,
                                           int               m,
                                           int               n,
                                           const float*      alpha,
                                           const float*      A,
                                           int               lda,
                                           hipblasStride     strideA,
                                           const float*      B,
                                           int               ldb,
                                           hipblasStride     strideB,
                                           const float*      beta,
                                           float*            C,
                                           int               ldc,
                                           hipblasStride     strideC,
                                           int               batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDsymmStridedBatched(hipblasHandle_t   handle,
                                           hipblasSideMode_t side,
                                           hipblasFillMode_t uplo,
                                           int               m,
                                           int               n,
                                           const double*     alpha,
                                           const double*     A,
                                           int               lda,
                                           hipblasStride     strideA,
                                           const double*     B,
                                           int               ldb,
                                           hipblasStride     strideB,
                                           const double*     beta,
                                           double*           C,
                                           int               ldc,
                                           hipblasStride     strideC,
                                           int               batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCsymmStridedBatched(hipblasHandle_t       handle,
                                           hipblasSideMode_t     side,
                                           hipblasFillMode_t     uplo,
                                           int                   m,
                                           int                   n,
                                           const hipblasComplex* alpha,
                                           const hipblasComplex* A,
                                           int                   lda,
                                           hipblasStride         strideA,
                                           const hipblasComplex* B,
                                           int                   ldb,
                                           hipblasStride         strideB,
                                           const hipblasComplex* beta,
                                           hipblasComplex*       C,
                                           int                   ldc,
                                           hipblasStride         strideC,
                                           int                   batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZsymmStridedBatched(hipblasHandle_t             handle,
                                           hipblasSideMode_t           side,
                                           hipblasFillMode_t           uplo,
                                           int                         m,
                                           int                         n,
                                           const hipblasDoubleComplex* alpha,
                                           const hipblasDoubleComplex* A,
                                           int                         lda,
                                           hipblasStride               strideA,
                                           const hipblasDoubleComplex* B,
                                           int                         ldb,
                                           hipblasStride               strideB,
                                           const hipblasDoubleComplex* beta,
                                           hipblasDoubleComplex*       C,
                                           int                         ldc,
                                           hipblasStride               strideC,
                                           int                         batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

// Level-3 : syrk(supported datatypes : float , double , float complex and double complex )
hipblasStatus_t hipblasSsyrk(hipblasHandle_t    handle,
                             hipblasFillMode_t  uplo,
                             hipblasOperation_t transA,
                             int                n,
                             int                k,
                             const float*       alpha,
                             const float*       A,
                             int                lda,
                             const float*       beta,
                             float*             C,
                             int                ldc)
try
{
    if (handle == nullptr || alpha == nullptr || A == nullptr || C == nullptr || beta == nullptr ||
        k <= 0 || n <= 0 || lda <= 0 || ldc <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    // As per spec alpha can be device/host memory. In case of device memory *alpha will crash
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    auto is_dev_ptr = (queryCurrentPtrMode(handle) == HIPBLAS_POINTER_MODE_DEVICE);

    float h_alpha, h_beta;
    if (is_dev_ptr) {
        hipMemcpy(&h_alpha, alpha, sizeof(float), hipMemcpyDefault);
        hipMemcpy(&h_beta, beta, sizeof(float), hipMemcpyDefault);
    } else {
        h_alpha = *((float*)alpha);
        h_beta = *((float*)beta);
    }

    onemklSsyrk(sycl_queue, convert(uplo), convert(transA), n, k, h_alpha, A, lda, h_beta, C, ldc);
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDsyrk(hipblasHandle_t    handle,
                             hipblasFillMode_t  uplo,
                             hipblasOperation_t transA,
                             int                n,
                             int                k,
                             const double*      alpha,
                             const double*      A,
                             int                lda,
                             const double*      beta,
                             double*            C,
                             int                ldc)
try
{
    if (handle == nullptr || alpha == nullptr || A == nullptr || C == nullptr || beta == nullptr ||
        k <= 0 || n <= 0 || lda <= 0 || ldc <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    // As per spec alpha can be device/host memory. In case of device memory *alpha will crash
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    auto is_dev_ptr = (queryCurrentPtrMode(handle) == HIPBLAS_POINTER_MODE_DEVICE);

    double h_alpha, h_beta;
    if (is_dev_ptr) {
        hipMemcpy(&h_alpha, alpha, sizeof(double), hipMemcpyDefault);
        hipMemcpy(&h_beta, beta, sizeof(double), hipMemcpyDefault);
    } else {
        h_alpha = *((double*)alpha);
        h_beta = *((double*)beta);
    }

    onemklDsyrk(sycl_queue, convert(uplo), convert(transA), n, k, h_alpha, A, lda, h_beta, C, ldc);
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCsyrk(hipblasHandle_t       handle,
                             hipblasFillMode_t     uplo,
                             hipblasOperation_t    transA,
                             int                   n,
                             int                   k,
                             const hipblasComplex* alpha,
                             const hipblasComplex* A,
                             int                   lda,
                             const hipblasComplex* beta,
                             hipblasComplex*       C,
                             int                   ldc)
try
{
    if (handle == nullptr || alpha == nullptr || A == nullptr || C == nullptr || beta == nullptr ||
        k <= 0 || n <= 0 || lda <= 0 || ldc <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    // As per spec alpha can be device/host memory. In case of device memory *alpha will crash
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    auto is_dev_ptr = (queryCurrentPtrMode(handle) == HIPBLAS_POINTER_MODE_DEVICE);

    float _Complex h_alpha, h_beta;
    if (is_dev_ptr) {
        hipMemcpy(&h_alpha, alpha, sizeof(float _Complex), hipMemcpyDefault);
        hipMemcpy(&h_beta, beta, sizeof(float _Complex), hipMemcpyDefault);
    } else {
        h_alpha = *((float _Complex*)alpha);
        h_beta = *((float _Complex*)beta);
    }

    onemklCsyrk(sycl_queue, convert(uplo), convert(transA), n, k, h_alpha, (const float _Complex*)A, lda,
                h_beta, (float _Complex*)C, ldc);
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZsyrk(hipblasHandle_t             handle,
                             hipblasFillMode_t           uplo,
                             hipblasOperation_t          transA,
                             int                         n,
                             int                         k,
                             const hipblasDoubleComplex* alpha,
                             const hipblasDoubleComplex* A,
                             int                         lda,
                             const hipblasDoubleComplex* beta,
                             hipblasDoubleComplex*       C,
                             int                         ldc)
try
{
    if (handle == nullptr || alpha == nullptr || A == nullptr || C == nullptr || beta == nullptr ||
        k <= 0 || n <= 0 || lda <= 0 || ldc <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    // As per spec alpha can be device/host memory. In case of device memory *alpha will crash
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    auto is_dev_ptr = (queryCurrentPtrMode(handle) == HIPBLAS_POINTER_MODE_DEVICE);

    double _Complex h_alpha, h_beta;
    if (is_dev_ptr) {
        hipMemcpy(&h_alpha, alpha, sizeof(double _Complex), hipMemcpyDefault);
        hipMemcpy(&h_beta, beta, sizeof(double _Complex), hipMemcpyDefault);
    } else {
        h_alpha = *((double _Complex*)alpha);
        h_beta = *((double _Complex*)beta);
    }

    onemklZsyrk(sycl_queue, convert(uplo), convert(transA), n, k, h_alpha, (const double _Complex*)A, lda,
                h_beta, (double _Complex*)C, ldc);
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

// syrk_batched
hipblasStatus_t hipblasSsyrkBatched(hipblasHandle_t    handle,
                                    hipblasFillMode_t  uplo,
                                    hipblasOperation_t transA,
                                    int                n,
                                    int                k,
                                    const float*       alpha,
                                    const float* const A[],
                                    int                lda,
                                    const float*       beta,
                                    float* const       C[],
                                    int                ldc,
                                    int                batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDsyrkBatched(hipblasHandle_t     handle,
                                    hipblasFillMode_t   uplo,
                                    hipblasOperation_t  transA,
                                    int                 n,
                                    int                 k,
                                    const double*       alpha,
                                    const double* const A[],
                                    int                 lda,
                                    const double*       beta,
                                    double* const       C[],
                                    int                 ldc,
                                    int                 batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCsyrkBatched(hipblasHandle_t             handle,
                                    hipblasFillMode_t           uplo,
                                    hipblasOperation_t          transA,
                                    int                         n,
                                    int                         k,
                                    const hipblasComplex*       alpha,
                                    const hipblasComplex* const A[],
                                    int                         lda,
                                    const hipblasComplex*       beta,
                                    hipblasComplex* const       C[],
                                    int                         ldc,
                                    int                         batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZsyrkBatched(hipblasHandle_t                   handle,
                                    hipblasFillMode_t                 uplo,
                                    hipblasOperation_t                transA,
                                    int                               n,
                                    int                               k,
                                    const hipblasDoubleComplex*       alpha,
                                    const hipblasDoubleComplex* const A[],
                                    int                               lda,
                                    const hipblasDoubleComplex*       beta,
                                    hipblasDoubleComplex* const       C[],
                                    int                               ldc,
                                    int                               batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

// syrk_strided_batched
hipblasStatus_t hipblasSsyrkStridedBatched(hipblasHandle_t    handle,
                                           hipblasFillMode_t  uplo,
                                           hipblasOperation_t transA,
                                           int                n,
                                           int                k,
                                           const float*       alpha,
                                           const float*       A,
                                           int                lda,
                                           hipblasStride      strideA,
                                           const float*       beta,
                                           float*             C,
                                           int                ldc,
                                           hipblasStride      strideC,
                                           int                batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDsyrkStridedBatched(hipblasHandle_t    handle,
                                           hipblasFillMode_t  uplo,
                                           hipblasOperation_t transA,
                                           int                n,
                                           int                k,
                                           const double*      alpha,
                                           const double*      A,
                                           int                lda,
                                           hipblasStride      strideA,
                                           const double*      beta,
                                           double*            C,
                                           int                ldc,
                                           hipblasStride      strideC,
                                           int                batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCsyrkStridedBatched(hipblasHandle_t       handle,
                                           hipblasFillMode_t     uplo,
                                           hipblasOperation_t    transA,
                                           int                   n,
                                           int                   k,
                                           const hipblasComplex* alpha,
                                           const hipblasComplex* A,
                                           int                   lda,
                                           hipblasStride         strideA,
                                           const hipblasComplex* beta,
                                           hipblasComplex*       C,
                                           int                   ldc,
                                           hipblasStride         strideC,
                                           int                   batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZsyrkStridedBatched(hipblasHandle_t             handle,
                                           hipblasFillMode_t           uplo,
                                           hipblasOperation_t          transA,
                                           int                         n,
                                           int                         k,
                                           const hipblasDoubleComplex* alpha,
                                           const hipblasDoubleComplex* A,
                                           int                         lda,
                                           hipblasStride               strideA,
                                           const hipblasDoubleComplex* beta,
                                           hipblasDoubleComplex*       C,
                                           int                         ldc,
                                           hipblasStride               strideC,
                                           int                         batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

// Level-3 : syr2k(supported datatypes : float , double , float complex and double complex )
hipblasStatus_t hipblasSsyr2k(hipblasHandle_t    handle,
                              hipblasFillMode_t  uplo,
                              hipblasOperation_t transA,
                              int                n,
                              int                k,
                              const float*       alpha,
                              const float*       A,
                              int                lda,
                              const float*       B,
                              int                ldb,
                              const float*       beta,
                              float*             C,
                              int                ldc)
try
{
    if (handle == nullptr || alpha == nullptr || A == nullptr || B == nullptr || C == nullptr || beta == nullptr ||
        k <= 0 || n <= 0 || lda <= 0 || ldb <= 0 || ldc <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    // As per spec alpha can be device/host memory. In case of device memory *alpha will crash
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    auto is_dev_ptr = (queryCurrentPtrMode(handle) == HIPBLAS_POINTER_MODE_DEVICE);

    float h_alpha, h_beta;
    if (is_dev_ptr) {
        hipMemcpy(&h_alpha, alpha, sizeof(float), hipMemcpyDefault);
        hipMemcpy(&h_beta, beta, sizeof(float), hipMemcpyDefault);
    } else {
        h_alpha = *((float*)alpha);
        h_beta = *((float*)beta);
    }

    onemklSsyr2k(sycl_queue, convert(uplo), convert(transA), n, k, h_alpha, A, lda, B, ldb, h_beta, C, ldc);
	return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDsyr2k(hipblasHandle_t    handle,
                              hipblasFillMode_t  uplo,
                              hipblasOperation_t transA,
                              int                n,
                              int                k,
                              const double*      alpha,
                              const double*      A,
                              int                lda,
                              const double*      B,
                              int                ldb,
                              const double*      beta,
                              double*            C,
                              int                ldc)
try
{
    if (handle == nullptr || alpha == nullptr || A == nullptr || B == nullptr || C == nullptr || beta == nullptr ||
        k <= 0 || n <= 0 || lda <= 0 || ldb <= 0 || ldc <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    // As per spec alpha can be device/host memory. In case of device memory *alpha will crash
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    auto is_dev_ptr = (queryCurrentPtrMode(handle) == HIPBLAS_POINTER_MODE_DEVICE);

    double h_alpha, h_beta;
    if (is_dev_ptr) {
        hipMemcpy(&h_alpha, alpha, sizeof(double), hipMemcpyDefault);
        hipMemcpy(&h_beta, beta, sizeof(double), hipMemcpyDefault);
    } else {
        h_alpha = *((double*)alpha);
        h_beta = *((double*)beta);
    }

    onemklDsyr2k(sycl_queue, convert(uplo), convert(transA), n, k, h_alpha, A, lda, B, ldb, h_beta, C, ldc);
	return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCsyr2k(hipblasHandle_t       handle,
                              hipblasFillMode_t     uplo,
                              hipblasOperation_t    transA,
                              int                   n,
                              int                   k,
                              const hipblasComplex* alpha,
                              const hipblasComplex* A,
                              int                   lda,
                              const hipblasComplex* B,
                              int                   ldb,
                              const hipblasComplex* beta,
                              hipblasComplex*       C,
                              int                   ldc)
try
{
    if (handle == nullptr || alpha == nullptr || A == nullptr || B == nullptr || C == nullptr || beta == nullptr ||
        k <= 0 || n <= 0 || lda <= 0 || ldb <= 0 || ldc <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    // As per spec alpha can be device/host memory. In case of device memory *alpha will crash
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    auto is_dev_ptr = (queryCurrentPtrMode(handle) == HIPBLAS_POINTER_MODE_DEVICE);

    float _Complex h_alpha, h_beta;
    if (is_dev_ptr) {
        hipMemcpy(&h_alpha, alpha, sizeof(float _Complex), hipMemcpyDefault);
        hipMemcpy(&h_beta, beta, sizeof(float _Complex), hipMemcpyDefault);
    } else {
        h_alpha = *((float _Complex*)alpha);
        h_beta = *((float _Complex*)beta);
    }

    onemklCsyr2k(sycl_queue, convert(uplo), convert(transA), n, k, h_alpha,
                 (const float _Complex*)A, lda, (const float _Complex*)B, ldb, h_beta, (float _Complex*)C, ldc);
	return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZsyr2k(hipblasHandle_t             handle,
                              hipblasFillMode_t           uplo,
                              hipblasOperation_t          transA,
                              int                         n,
                              int                         k,
                              const hipblasDoubleComplex* alpha,
                              const hipblasDoubleComplex* A,
                              int                         lda,
                              const hipblasDoubleComplex* B,
                              int                         ldb,
                              const hipblasDoubleComplex* beta,
                              hipblasDoubleComplex*       C,
                              int                         ldc)
try
{
    if (handle == nullptr || alpha == nullptr || A == nullptr || B == nullptr || C == nullptr || beta == nullptr ||
        k <= 0 || n <= 0 || lda <= 0 || ldb <= 0 || ldc <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    // As per spec alpha can be device/host memory. In case of device memory *alpha will crash
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    auto is_dev_ptr = (queryCurrentPtrMode(handle) == HIPBLAS_POINTER_MODE_DEVICE);

    double _Complex h_alpha, h_beta;
    if (is_dev_ptr) {
        hipMemcpy(&h_alpha, alpha, sizeof(double _Complex), hipMemcpyDefault);
        hipMemcpy(&h_beta, beta, sizeof(double _Complex), hipMemcpyDefault);
    } else {
        h_alpha = *((double _Complex*)alpha);
        h_beta = *((double _Complex*)beta);
    }

    onemklZsyr2k(sycl_queue, convert(uplo), convert(transA), n, k, h_alpha,
                 (const double _Complex*)A, lda, (const double _Complex*)B, ldb, h_beta, (double _Complex*)C, ldc);
	return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

// syr2k_batched
hipblasStatus_t hipblasSsyr2kBatched(hipblasHandle_t    handle,
                                     hipblasFillMode_t  uplo,
                                     hipblasOperation_t transA,
                                     int                n,
                                     int                k,
                                     const float*       alpha,
                                     const float* const A[],
                                     int                lda,
                                     const float* const B[],
                                     int                ldb,
                                     const float*       beta,
                                     float* const       C[],
                                     int                ldc,
                                     int                batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDsyr2kBatched(hipblasHandle_t     handle,
                                     hipblasFillMode_t   uplo,
                                     hipblasOperation_t  transA,
                                     int                 n,
                                     int                 k,
                                     const double*       alpha,
                                     const double* const A[],
                                     int                 lda,
                                     const double* const B[],
                                     int                 ldb,
                                     const double*       beta,
                                     double* const       C[],
                                     int                 ldc,
                                     int                 batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCsyr2kBatched(hipblasHandle_t             handle,
                                     hipblasFillMode_t           uplo,
                                     hipblasOperation_t          transA,
                                     int                         n,
                                     int                         k,
                                     const hipblasComplex*       alpha,
                                     const hipblasComplex* const A[],
                                     int                         lda,
                                     const hipblasComplex* const B[],
                                     int                         ldb,
                                     const hipblasComplex*       beta,
                                     hipblasComplex* const       C[],
                                     int                         ldc,
                                     int                         batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZsyr2kBatched(hipblasHandle_t                   handle,
                                     hipblasFillMode_t                 uplo,
                                     hipblasOperation_t                transA,
                                     int                               n,
                                     int                               k,
                                     const hipblasDoubleComplex*       alpha,
                                     const hipblasDoubleComplex* const A[],
                                     int                               lda,
                                     const hipblasDoubleComplex* const B[],
                                     int                               ldb,
                                     const hipblasDoubleComplex*       beta,
                                     hipblasDoubleComplex* const       C[],
                                     int                               ldc,
                                     int                               batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

// syr2k_strided_batched
hipblasStatus_t hipblasSsyr2kStridedBatched(hipblasHandle_t    handle,
                                            hipblasFillMode_t  uplo,
                                            hipblasOperation_t transA,
                                            int                n,
                                            int                k,
                                            const float*       alpha,
                                            const float*       A,
                                            int                lda,
                                            hipblasStride      strideA,
                                            const float*       B,
                                            int                ldb,
                                            hipblasStride      strideB,
                                            const float*       beta,
                                            float*             C,
                                            int                ldc,
                                            hipblasStride      strideC,
                                            int                batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDsyr2kStridedBatched(hipblasHandle_t    handle,
                                            hipblasFillMode_t  uplo,
                                            hipblasOperation_t transA,
                                            int                n,
                                            int                k,
                                            const double*      alpha,
                                            const double*      A,
                                            int                lda,
                                            hipblasStride      strideA,
                                            const double*      B,
                                            int                ldb,
                                            hipblasStride      strideB,
                                            const double*      beta,
                                            double*            C,
                                            int                ldc,
                                            hipblasStride      strideC,
                                            int                batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCsyr2kStridedBatched(hipblasHandle_t       handle,
                                            hipblasFillMode_t     uplo,
                                            hipblasOperation_t    transA,
                                            int                   n,
                                            int                   k,
                                            const hipblasComplex* alpha,
                                            const hipblasComplex* A,
                                            int                   lda,
                                            hipblasStride         strideA,
                                            const hipblasComplex* B,
                                            int                   ldb,
                                            hipblasStride         strideB,
                                            const hipblasComplex* beta,
                                            hipblasComplex*       C,
                                            int                   ldc,
                                            hipblasStride         strideC,
                                            int                   batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZsyr2kStridedBatched(hipblasHandle_t             handle,
                                            hipblasFillMode_t           uplo,
                                            hipblasOperation_t          transA,
                                            int                         n,
                                            int                         k,
                                            const hipblasDoubleComplex* alpha,
                                            const hipblasDoubleComplex* A,
                                            int                         lda,
                                            hipblasStride               strideA,
                                            const hipblasDoubleComplex* B,
                                            int                         ldb,
                                            hipblasStride               strideB,
                                            const hipblasDoubleComplex* beta,
                                            hipblasDoubleComplex*       C,
                                            int                         ldc,
                                            hipblasStride               strideC,
                                            int                         batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

// Level-3 : hemm(supported datatypes : float complex and double complex )
hipblasStatus_t hipblasChemm(hipblasHandle_t       handle,
                             hipblasSideMode_t     side,
                             hipblasFillMode_t     uplo,
                             int                   n,
                             int                   k,
                             const hipblasComplex* alpha,
                             const hipblasComplex* A,
                             int                   lda,
                             const hipblasComplex* B,
                             int                   ldb,
                             const hipblasComplex* beta,
                             hipblasComplex*       C,
                             int                   ldc)
try
{
    if (handle == nullptr || alpha == nullptr || A == nullptr || B == nullptr || C == nullptr || beta == nullptr ||
        k <= 0 || n <= 0 || lda <= 0 || ldb <= 0 || ldc <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    // As per spec alpha can be device/host memory. In case of device memory *alpha will crash
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    auto is_dev_ptr = (queryCurrentPtrMode(handle) == HIPBLAS_POINTER_MODE_DEVICE);

    float _Complex h_alpha, h_beta;
    if (is_dev_ptr) {
        hipMemcpy(&h_alpha, alpha, sizeof(float _Complex), hipMemcpyDefault);
        hipMemcpy(&h_beta, beta, sizeof(float _Complex), hipMemcpyDefault);
    } else {
        h_alpha = *((float _Complex*)alpha);
        h_beta = *((float _Complex*)beta);
    }

    onemklChemm(sycl_queue, convert(side), convert(uplo), n, k, h_alpha, (const float _Complex*)A, lda, 
                (const float _Complex*)B, ldb, h_beta, (float _Complex*)C, ldc);
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZhemm(hipblasHandle_t             handle,
                             hipblasSideMode_t           side,
                             hipblasFillMode_t           uplo,
                             int                         n,
                             int                         k,
                             const hipblasDoubleComplex* alpha,
                             const hipblasDoubleComplex* A,
                             int                         lda,
                             const hipblasDoubleComplex* B,
                             int                         ldb,
                             const hipblasDoubleComplex* beta,
                             hipblasDoubleComplex*       C,
                             int                         ldc)
try
{
    if (handle == nullptr || alpha == nullptr || A == nullptr || B == nullptr || C == nullptr || beta == nullptr ||
        k <= 0 || n <= 0 || lda <= 0 || ldb <= 0 || ldc <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    // As per spec alpha can be device/host memory. In case of device memory *alpha will crash
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    auto is_dev_ptr = (queryCurrentPtrMode(handle) == HIPBLAS_POINTER_MODE_DEVICE);

    double _Complex h_alpha, h_beta;
    if (is_dev_ptr) {
        hipMemcpy(&h_alpha, alpha, sizeof(double _Complex), hipMemcpyDefault);
        hipMemcpy(&h_beta, beta, sizeof(double _Complex), hipMemcpyDefault);
    } else {
        h_alpha = *((double _Complex*)alpha);
        h_beta = *((double _Complex*)beta);
    }

    onemklZhemm(sycl_queue, convert(side), convert(uplo), n, k, h_alpha, (const double _Complex*)A, lda, 
                (const double _Complex*)B, ldb, h_beta, (double _Complex*)C, ldc);
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

// hemm_batched
hipblasStatus_t hipblasChemmBatched(hipblasHandle_t             handle,
                                    hipblasSideMode_t           side,
                                    hipblasFillMode_t           uplo,
                                    int                         n,
                                    int                         k,
                                    const hipblasComplex*       alpha,
                                    const hipblasComplex* const A[],
                                    int                         lda,
                                    const hipblasComplex* const B[],
                                    int                         ldb,
                                    const hipblasComplex*       beta,
                                    hipblasComplex* const       C[],
                                    int                         ldc,
                                    int                         batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZhemmBatched(hipblasHandle_t                   handle,
                                    hipblasSideMode_t                 side,
                                    hipblasFillMode_t                 uplo,
                                    int                               n,
                                    int                               k,
                                    const hipblasDoubleComplex*       alpha,
                                    const hipblasDoubleComplex* const A[],
                                    int                               lda,
                                    const hipblasDoubleComplex* const B[],
                                    int                               ldb,
                                    const hipblasDoubleComplex*       beta,
                                    hipblasDoubleComplex* const       C[],
                                    int                               ldc,
                                    int                               batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

// hemm_strided_batched
hipblasStatus_t hipblasChemmStridedBatched(hipblasHandle_t       handle,
                                           hipblasSideMode_t     side,
                                           hipblasFillMode_t     uplo,
                                           int                   n,
                                           int                   k,
                                           const hipblasComplex* alpha,
                                           const hipblasComplex* A,
                                           int                   lda,
                                           hipblasStride         strideA,
                                           const hipblasComplex* B,
                                           int                   ldb,
                                           hipblasStride         strideB,
                                           const hipblasComplex* beta,
                                           hipblasComplex*       C,
                                           int                   ldc,
                                           hipblasStride         strideC,
                                           int                   batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZhemmStridedBatched(hipblasHandle_t             handle,
                                           hipblasSideMode_t           side,
                                           hipblasFillMode_t           uplo,
                                           int                         n,
                                           int                         k,
                                           const hipblasDoubleComplex* alpha,
                                           const hipblasDoubleComplex* A,
                                           int                         lda,
                                           hipblasStride               strideA,
                                           const hipblasDoubleComplex* B,
                                           int                         ldb,
                                           hipblasStride               strideB,
                                           const hipblasDoubleComplex* beta,
                                           hipblasDoubleComplex*       C,
                                           int                         ldc,
                                           hipblasStride               strideC,
                                           int                         batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

// Level-3 : trmm(supported datatypes : float , double , float complex and double complex )
hipblasStatus_t hipblasStrmm(hipblasHandle_t    handle,
                             hipblasSideMode_t  side,
                             hipblasFillMode_t  uplo,
                             hipblasOperation_t transA,
                             hipblasDiagType_t  diag,
                             int                m,
                             int                n,
                             const float*       alpha,
                             const float*       A,
                             int                lda,
                             float*             B,
                             int                ldb)
try
{
    if (handle == nullptr || alpha == nullptr || A == nullptr || B == nullptr ||
        m <= 0 || n <= 0 || lda <= 0 || ldb <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    // As per spec alpha can be device/host memory. In case of device memory *alpha will crash
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    auto is_dev_ptr = (queryCurrentPtrMode(handle) == HIPBLAS_POINTER_MODE_DEVICE);

    float h_alpha;
    if (is_dev_ptr) {
        hipMemcpy(&h_alpha, alpha, sizeof(float), hipMemcpyDefault);
    } else {
        h_alpha = *((float*)alpha);
    }
    onemklStrmm(sycl_queue, convert(side), convert(uplo), convert(transA), convert(diag), m, n, h_alpha, A, lda, B, ldb);
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDtrmm(hipblasHandle_t    handle,
                             hipblasSideMode_t  side,
                             hipblasFillMode_t  uplo,
                             hipblasOperation_t transA,
                             hipblasDiagType_t  diag,
                             int                m,
                             int                n,
                             const double*      alpha,
                             const double*      A,
                             int                lda,
                             double*            B,
                             int                ldb)
try
{
    if (handle == nullptr || alpha == nullptr || A == nullptr || B == nullptr ||
        m <= 0 || n <= 0 || lda <= 0 || ldb <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    // As per spec alpha can be device/host memory. In case of device memory *alpha will crash
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    auto is_dev_ptr = (queryCurrentPtrMode(handle) == HIPBLAS_POINTER_MODE_DEVICE);

    double h_alpha;
    if (is_dev_ptr) {
        hipMemcpy(&h_alpha, alpha, sizeof(double), hipMemcpyDefault);
    } else {
        h_alpha = *((double*)alpha);
    }
    onemklDtrmm(sycl_queue, convert(side), convert(uplo), convert(transA), convert(diag), m, n, h_alpha, A, lda, B, ldb);
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCtrmm(hipblasHandle_t       handle,
                             hipblasSideMode_t     side,
                             hipblasFillMode_t     uplo,
                             hipblasOperation_t    transA,
                             hipblasDiagType_t     diag,
                             int                   m,
                             int                   n,
                             const hipblasComplex* alpha,
                             const hipblasComplex* A,
                             int                   lda,
                             hipblasComplex*       B,
                             int                   ldb)
try
{
    if (handle == nullptr || alpha == nullptr || A == nullptr || B == nullptr ||
        m <= 0 || n <= 0 || lda <= 0 || ldb <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    // As per spec alpha can be device/host memory. In case of device memory *alpha will crash
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    auto is_dev_ptr = (queryCurrentPtrMode(handle) == HIPBLAS_POINTER_MODE_DEVICE);

    float _Complex h_alpha;
    if (is_dev_ptr) {
        hipMemcpy(&h_alpha, alpha, sizeof(float _Complex), hipMemcpyDefault);
    } else {
        h_alpha = *((float _Complex*)alpha);
    }
    onemklCtrmm(sycl_queue, convert(side), convert(uplo), convert(transA), convert(diag), m, n, h_alpha,
                (const float _Complex*)A, lda, (float _Complex*)B, ldb);
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZtrmm(hipblasHandle_t             handle,
                             hipblasSideMode_t           side,
                             hipblasFillMode_t           uplo,
                             hipblasOperation_t          transA,
                             hipblasDiagType_t           diag,
                             int                         m,
                             int                         n,
                             const hipblasDoubleComplex* alpha,
                             const hipblasDoubleComplex* A,
                             int                         lda,
                             hipblasDoubleComplex*       B,
                             int                         ldb)
try
{
    if (handle == nullptr || alpha == nullptr || A == nullptr || B == nullptr ||
        m <= 0 || n <= 0 || lda <= 0 || ldb <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    // As per spec alpha can be device/host memory. In case of device memory *alpha will crash
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    auto is_dev_ptr = (queryCurrentPtrMode(handle) == HIPBLAS_POINTER_MODE_DEVICE);

    double _Complex h_alpha;
    if (is_dev_ptr) {
        hipMemcpy(&h_alpha, alpha, sizeof(double _Complex), hipMemcpyDefault);
    } else {
        h_alpha = *((double _Complex*)alpha);
    }
    onemklZtrmm(sycl_queue, convert(side), convert(uplo), convert(transA), convert(diag), m, n, h_alpha,
                (const double _Complex*)A, lda, (double _Complex*)B, ldb);
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

// trmm_batched
hipblasStatus_t hipblasStrmmBatched(hipblasHandle_t    handle,
                                    hipblasSideMode_t  side,
                                    hipblasFillMode_t  uplo,
                                    hipblasOperation_t transA,
                                    hipblasDiagType_t  diag,
                                    int                m,
                                    int                n,
                                    const float*       alpha,
                                    const float* const A[],
                                    int                lda,
                                    float* const       B[],
                                    int                ldb,
                                    int                batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDtrmmBatched(hipblasHandle_t     handle,
                                    hipblasSideMode_t   side,
                                    hipblasFillMode_t   uplo,
                                    hipblasOperation_t  transA,
                                    hipblasDiagType_t   diag,
                                    int                 m,
                                    int                 n,
                                    const double*       alpha,
                                    const double* const A[],
                                    int                 lda,
                                    double* const       B[],
                                    int                 ldb,
                                    int                 batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCtrmmBatched(hipblasHandle_t             handle,
                                    hipblasSideMode_t           side,
                                    hipblasFillMode_t           uplo,
                                    hipblasOperation_t          transA,
                                    hipblasDiagType_t           diag,
                                    int                         m,
                                    int                         n,
                                    const hipblasComplex*       alpha,
                                    const hipblasComplex* const A[],
                                    int                         lda,
                                    hipblasComplex* const       B[],
                                    int                         ldb,
                                    int                         batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZtrmmBatched(hipblasHandle_t                   handle,
                                    hipblasSideMode_t                 side,
                                    hipblasFillMode_t                 uplo,
                                    hipblasOperation_t                transA,
                                    hipblasDiagType_t                 diag,
                                    int                               m,
                                    int                               n,
                                    const hipblasDoubleComplex*       alpha,
                                    const hipblasDoubleComplex* const A[],
                                    int                               lda,
                                    hipblasDoubleComplex* const       B[],
                                    int                               ldb,
                                    int                               batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

// trmm_strided_batched
hipblasStatus_t hipblasStrmmStridedBatched(hipblasHandle_t    handle,
                                           hipblasSideMode_t  side,
                                           hipblasFillMode_t  uplo,
                                           hipblasOperation_t transA,
                                           hipblasDiagType_t  diag,
                                           int                m,
                                           int                n,
                                           const float*       alpha,
                                           const float*       A,
                                           int                lda,
                                           hipblasStride      strideA,
                                           float*             B,
                                           int                ldb,
                                           hipblasStride      strideB,
                                           int                batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDtrmmStridedBatched(hipblasHandle_t    handle,
                                           hipblasSideMode_t  side,
                                           hipblasFillMode_t  uplo,
                                           hipblasOperation_t transA,
                                           hipblasDiagType_t  diag,
                                           int                m,
                                           int                n,
                                           const double*      alpha,
                                           const double*      A,
                                           int                lda,
                                           hipblasStride      strideA,
                                           double*            B,
                                           int                ldb,
                                           hipblasStride      strideB,
                                           int                batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCtrmmStridedBatched(hipblasHandle_t       handle,
                                           hipblasSideMode_t     side,
                                           hipblasFillMode_t     uplo,
                                           hipblasOperation_t    transA,
                                           hipblasDiagType_t     diag,
                                           int                   m,
                                           int                   n,
                                           const hipblasComplex* alpha,
                                           const hipblasComplex* A,
                                           int                   lda,
                                           hipblasStride         strideA,
                                           hipblasComplex*       B,
                                           int                   ldb,
                                           hipblasStride         strideB,
                                           int                   batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZtrmmStridedBatched(hipblasHandle_t             handle,
                                           hipblasSideMode_t           side,
                                           hipblasFillMode_t           uplo,
                                           hipblasOperation_t          transA,
                                           hipblasDiagType_t           diag,
                                           int                         m,
                                           int                         n,
                                           const hipblasDoubleComplex* alpha,
                                           const hipblasDoubleComplex* A,
                                           int                         lda,
                                           hipblasStride               strideA,
                                           hipblasDoubleComplex*       B,
                                           int                         ldb,
                                           hipblasStride               strideB,
                                           int                         batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

// Level-3 : trsm(supported datatypes : float , double , float complex and double complex )
hipblasStatus_t hipblasStrsm(hipblasHandle_t    handle,
                             hipblasSideMode_t  side,
                             hipblasFillMode_t  uplo,
                             hipblasOperation_t transA,
                             hipblasDiagType_t  diag,
                             int                m,
                             int                n,
                             const float*       alpha,
                             float*             A,
                             int                lda,
                             float*             B,
                             int                ldb)
try
{
    if (handle == nullptr || alpha == nullptr || A == nullptr || B == nullptr ||
        m <= 0 || n <= 0 || lda <= 0 || ldb <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    // As per spec alpha can be device/host memory. In case of device memory *alpha will crash
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    auto is_dev_ptr = (queryCurrentPtrMode(handle) == HIPBLAS_POINTER_MODE_DEVICE);

    float h_alpha;
    if (is_dev_ptr) {
        hipMemcpy(&h_alpha, alpha, sizeof(float), hipMemcpyDefault);
    } else {
        h_alpha = *((float*)alpha);
    }
    onemklStrsm(sycl_queue, convert(side), convert(uplo), convert(transA), convert(diag), m, n, h_alpha, A, lda, B, ldb);
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDtrsm(hipblasHandle_t    handle,
                             hipblasSideMode_t  side,
                             hipblasFillMode_t  uplo,
                             hipblasOperation_t transA,
                             hipblasDiagType_t  diag,
                             int                m,
                             int                n,
                             const double*      alpha,
                             double*            A,
                             int                lda,
                             double*            B,
                             int                ldb)
try
{
    if (handle == nullptr || alpha == nullptr || A == nullptr || B == nullptr ||
        m <= 0 || n <= 0 || lda <= 0 || ldb <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    // As per spec alpha can be device/host memory. In case of device memory *alpha will crash
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    auto is_dev_ptr = (queryCurrentPtrMode(handle) == HIPBLAS_POINTER_MODE_DEVICE);

    double h_alpha;
    if (is_dev_ptr) {
        hipMemcpy(&h_alpha, alpha, sizeof(double), hipMemcpyDefault);
    } else {
        h_alpha = *((double*)alpha);
    }
    onemklDtrsm(sycl_queue, convert(side), convert(uplo), convert(transA), convert(diag), m, n, h_alpha, A, lda, B, ldb);
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCtrsm(hipblasHandle_t       handle,
                             hipblasSideMode_t     side,
                             hipblasFillMode_t     uplo,
                             hipblasOperation_t    transA,
                             hipblasDiagType_t     diag,
                             int                   m,
                             int                   n,
                             const hipblasComplex* alpha,
                             hipblasComplex*       A,
                             int                   lda,
                             hipblasComplex*       B,
                             int                   ldb)
try
{
    if (handle == nullptr || alpha == nullptr || A == nullptr || B == nullptr ||
        m <= 0 || n <= 0 || lda <= 0 || ldb <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    // As per spec alpha can be device/host memory. In case of device memory *alpha will crash
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    auto is_dev_ptr = (queryCurrentPtrMode(handle) == HIPBLAS_POINTER_MODE_DEVICE);

    float _Complex h_alpha;
    if (is_dev_ptr) {
        hipMemcpy(&h_alpha, alpha, sizeof(float _Complex), hipMemcpyDefault);
    } else {
        h_alpha = *((float _Complex*)alpha);
    }
    onemklCtrsm(sycl_queue, convert(side), convert(uplo), convert(transA), convert(diag), m, n,
                h_alpha, (const float _Complex*)A, lda, (float _Complex*)B, ldb);
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZtrsm(hipblasHandle_t             handle,
                             hipblasSideMode_t           side,
                             hipblasFillMode_t           uplo,
                             hipblasOperation_t          transA,
                             hipblasDiagType_t           diag,
                             int                         m,
                             int                         n,
                             const hipblasDoubleComplex* alpha,
                             hipblasDoubleComplex*       A,
                             int                         lda,
                             hipblasDoubleComplex*       B,
                             int                         ldb)
try
{
    if (handle == nullptr || alpha == nullptr || A == nullptr || B == nullptr ||
        m <= 0 || n <= 0 || lda <= 0 || ldb <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    // As per spec alpha can be device/host memory. In case of device memory *alpha will crash
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    auto is_dev_ptr = (queryCurrentPtrMode(handle) == HIPBLAS_POINTER_MODE_DEVICE);

    double _Complex h_alpha;
    if (is_dev_ptr) {
        hipMemcpy(&h_alpha, alpha, sizeof(double _Complex), hipMemcpyDefault);
    } else {
        h_alpha = *((double _Complex*)alpha);
    }
    onemklZtrsm(sycl_queue, convert(side), convert(uplo), convert(transA), convert(diag), m, n,
                h_alpha, (const double _Complex*)A, lda, (double _Complex*)B, ldb);
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

// trsm_batched
hipblasStatus_t hipblasStrsmBatched(hipblasHandle_t    handle,
                                    hipblasSideMode_t  side,
                                    hipblasFillMode_t  uplo,
                                    hipblasOperation_t transA,
                                    hipblasDiagType_t  diag,
                                    int                m,
                                    int                n,
                                    const float*       alpha,
                                    float* const       A[],
                                    int                lda,
                                    float*             B[],
                                    int                ldb,
                                    int                batch_count)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDtrsmBatched(hipblasHandle_t    handle,
                                    hipblasSideMode_t  side,
                                    hipblasFillMode_t  uplo,
                                    hipblasOperation_t transA,
                                    hipblasDiagType_t  diag,
                                    int                m,
                                    int                n,
                                    const double*      alpha,
                                    double* const      A[],
                                    int                lda,
                                    double*            B[],
                                    int                ldb,
                                    int                batch_count)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCtrsmBatched(hipblasHandle_t       handle,
                                    hipblasSideMode_t     side,
                                    hipblasFillMode_t     uplo,
                                    hipblasOperation_t    transA,
                                    hipblasDiagType_t     diag,
                                    int                   m,
                                    int                   n,
                                    const hipblasComplex* alpha,
                                    hipblasComplex* const A[],
                                    int                   lda,
                                    hipblasComplex*       B[],
                                    int                   ldb,
                                    int                   batch_count)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZtrsmBatched(hipblasHandle_t             handle,
                                    hipblasSideMode_t           side,
                                    hipblasFillMode_t           uplo,
                                    hipblasOperation_t          transA,
                                    hipblasDiagType_t           diag,
                                    int                         m,
                                    int                         n,
                                    const hipblasDoubleComplex* alpha,
                                    hipblasDoubleComplex* const A[],
                                    int                         lda,
                                    hipblasDoubleComplex*       B[],
                                    int                         ldb,
                                    int                         batch_count)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

// trsm_strided_batched
hipblasStatus_t hipblasStrsmStridedBatched(hipblasHandle_t    handle,
                                           hipblasSideMode_t  side,
                                           hipblasFillMode_t  uplo,
                                           hipblasOperation_t transA,
                                           hipblasDiagType_t  diag,
                                           int                m,
                                           int                n,
                                           const float*       alpha,
                                           float*             A,
                                           int                lda,
                                           hipblasStride      strideA,
                                           float*             B,
                                           int                ldb,
                                           hipblasStride      strideB,
                                           int                batch_count)
try
{
    if (handle == nullptr || alpha == nullptr || A == nullptr || B == nullptr ||
        m <= 0 || n <= 0 || lda <= 0 || ldb <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    // As per spec alpha can be device/host memory. In case of device memory *alpha will crash
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    auto is_dev_ptr = (queryCurrentPtrMode(handle) == HIPBLAS_POINTER_MODE_DEVICE);

    float h_alpha;
    if (is_dev_ptr) {
        hipMemcpy(&h_alpha, alpha, sizeof(float), hipMemcpyDefault);
    } else {
        h_alpha = *((float*)alpha);
    }
    onemklStrsm_strided(sycl_queue, convert(side), convert(uplo), convert(transA), convert(diag), m, n, h_alpha, A, lda, strideA, B, ldb, strideB, batch_count);
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDtrsmStridedBatched(hipblasHandle_t    handle,
                                           hipblasSideMode_t  side,
                                           hipblasFillMode_t  uplo,
                                           hipblasOperation_t transA,
                                           hipblasDiagType_t  diag,
                                           int                m,
                                           int                n,
                                           const double*      alpha,
                                           double*            A,
                                           int                lda,
                                           hipblasStride      strideA,
                                           double*            B,
                                           int                ldb,
                                           hipblasStride      strideB,
                                           int                batch_count)
try
{
    if (handle == nullptr || alpha == nullptr || A == nullptr || B == nullptr ||
        m <= 0 || n <= 0 || lda <= 0 || ldb <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    // As per spec alpha can be device/host memory. In case of device memory *alpha will crash
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    auto is_dev_ptr = (queryCurrentPtrMode(handle) == HIPBLAS_POINTER_MODE_DEVICE);

    double h_alpha;
    if (is_dev_ptr) {
        hipMemcpy(&h_alpha, alpha, sizeof(double), hipMemcpyDefault);
    } else {
        h_alpha = *((double*)alpha);
    }
    onemklDtrsm_strided(sycl_queue, convert(side), convert(uplo), convert(transA), convert(diag),
                                    m, n, h_alpha, A, lda, strideA, B, ldb, strideB, batch_count);
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCtrsmStridedBatched(hipblasHandle_t       handle,
                                           hipblasSideMode_t     side,
                                           hipblasFillMode_t     uplo,
                                           hipblasOperation_t    transA,
                                           hipblasDiagType_t     diag,
                                           int                   m,
                                           int                   n,
                                           const hipblasComplex* alpha,
                                           hipblasComplex*       A,
                                           int                   lda,
                                           hipblasStride         strideA,
                                           hipblasComplex*       B,
                                           int                   ldb,
                                           hipblasStride         strideB,
                                           int                   batch_count)
try
{
    if (handle == nullptr || alpha == nullptr || A == nullptr || B == nullptr ||
        m <= 0 || n <= 0 || lda <= 0 || ldb <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    // As per spec alpha can be device/host memory. In case of device memory *alpha will crash
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    auto is_dev_ptr = (queryCurrentPtrMode(handle) == HIPBLAS_POINTER_MODE_DEVICE);

    float _Complex h_alpha;
    if (is_dev_ptr) {
        hipMemcpy(&h_alpha, alpha, sizeof(float _Complex), hipMemcpyDefault);
    } else {
        h_alpha = *((float _Complex*)alpha);
    }
    onemklCtrsm_strided(sycl_queue, convert(side), convert(uplo), convert(transA), convert(diag), m, n,
                h_alpha, (const float _Complex*)A, lda, strideA, (float _Complex*)B, ldb, strideB, batch_count);
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZtrsmStridedBatched(hipblasHandle_t             handle,
                                           hipblasSideMode_t           side,
                                           hipblasFillMode_t           uplo,
                                           hipblasOperation_t          transA,
                                           hipblasDiagType_t           diag,
                                           int                         m,
                                           int                         n,
                                           const hipblasDoubleComplex* alpha,
                                           hipblasDoubleComplex*       A,
                                           int                         lda,
                                           hipblasStride               strideA,
                                           hipblasDoubleComplex*       B,
                                           int                         ldb,
                                           hipblasStride               strideB,
                                           int                         batch_count)
try
{
    if (handle == nullptr || alpha == nullptr || A == nullptr || B == nullptr ||
        m <= 0 || n <= 0 || lda <= 0 || ldb <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    // As per spec alpha can be device/host memory. In case of device memory *alpha will crash
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    auto is_dev_ptr = (queryCurrentPtrMode(handle) == HIPBLAS_POINTER_MODE_DEVICE);

    double _Complex h_alpha;
    if (is_dev_ptr) {
        hipMemcpy(&h_alpha, alpha, sizeof(double _Complex), hipMemcpyDefault);
    } else {
        h_alpha = *((double _Complex*)alpha);
    }
    onemklZtrsm_strided(sycl_queue, convert(side), convert(uplo), convert(transA), convert(diag), m, n,
                h_alpha, (const double _Complex*)A, lda, strideA, (double _Complex*)B, ldb, strideB, batch_count);
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

// Level-3 : gemm(supported datatypes : half, float , double , float complex and double complex )
hipblasStatus_t hipblasHgemm(hipblasHandle_t    handle,
                             hipblasOperation_t transa,
                             hipblasOperation_t transb,
                             int                m,
                             int                n,
                             int                k,
                             const hipblasHalf* alpha,
                             const hipblasHalf* A,
                             int                lda,
                             const hipblasHalf* B,
                             int                ldb,
                             const hipblasHalf* beta,
                             hipblasHalf*       C,
                             int                ldc)
try
{
    if (handle == nullptr || alpha == nullptr || A == nullptr || B == nullptr || C == nullptr || beta == nullptr ||
        m <= 0 || n <= 0 || k <= 0 || lda <= 0 || ldb <= 0 || ldc <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    // As per spec alpha can be device/host memory. In case of device memory *alpha will crash
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    auto is_dev_ptr = (queryCurrentPtrMode(handle) == HIPBLAS_POINTER_MODE_DEVICE);

    uint16_t h_alpha, h_beta;
    if (is_dev_ptr) {
        hipMemcpy(&h_alpha, alpha, sizeof(float), hipMemcpyDefault);
        hipMemcpy(&h_beta, beta, sizeof(float), hipMemcpyDefault);
    } else {
        h_alpha = *((float*)alpha);
        h_beta = *((float*)beta);
    }

    onemklHgemm(sycl_queue, convert(transa), convert(transb), m, n, k,
                h_alpha, (const short*)A, lda, (const short*)B, ldb, h_beta, (short *)C, ldc);
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasSgemm(hipblasHandle_t    handle,
                             hipblasOperation_t transa,
                             hipblasOperation_t transb,
                             int                m,
                             int                n,
                             int                k,
                             const float*       alpha,
                             const float*       A,
                             int                lda,
                             const float*       B,
                             int                ldb,
                             const float*       beta,
                             float*             C,
                             int                ldc)
try
{
    if (handle == nullptr || alpha == nullptr || A == nullptr || B == nullptr || C == nullptr || beta == nullptr ||
        m <= 0 || n <= 0 || k <= 0 || lda <= 0 || ldb <= 0 || ldc <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    // As per spec alpha can be device/host memory. In case of device memory *alpha will crash
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    auto is_dev_ptr = (queryCurrentPtrMode(handle) == HIPBLAS_POINTER_MODE_DEVICE);

    float h_alpha, h_beta;
    if (is_dev_ptr) {
        hipMemcpy(&h_alpha, alpha, sizeof(float), hipMemcpyDefault);
        hipMemcpy(&h_beta, beta, sizeof(float), hipMemcpyDefault);
    } else {
        h_alpha = *((float*)alpha);
        h_beta = *((float*)beta);
    }

    onemklSgemm(sycl_queue, convert(transa), convert(transb), m, n, k,
                h_alpha, A, lda, B, ldb, h_beta, C, ldc);
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDgemm(hipblasHandle_t    handle,
                             hipblasOperation_t transa,
                             hipblasOperation_t transb,
                             int                m,
                             int                n,
                             int                k,
                             const double*      alpha,
                             const double*      A,
                             int                lda,
                             const double*      B,
                             int                ldb,
                             const double*      beta,
                             double*            C,
                             int                ldc)
try
{
    if (handle == nullptr || alpha == nullptr || A == nullptr || B == nullptr || C == nullptr || beta == nullptr ||
        m <= 0 || n <= 0 || k <= 0 || lda <= 0 || ldb <= 0 || ldc <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    // As per spec alpha can be device/host memory. In case of device memory *alpha will crash
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    auto is_dev_ptr = (queryCurrentPtrMode(handle) == HIPBLAS_POINTER_MODE_DEVICE);

    double h_alpha, h_beta;
    if (is_dev_ptr) {
        hipMemcpy(&h_alpha, alpha, sizeof(double), hipMemcpyDefault);
        hipMemcpy(&h_beta, beta, sizeof(double), hipMemcpyDefault);
    } else {
        h_alpha = *((double*)alpha);
        h_beta = *((double*)beta);
    }

    onemklDgemm(sycl_queue, convert(transa), convert(transb), m, n, k,
                h_alpha, A, lda, B, ldb, h_beta, C, ldc);
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCgemm(hipblasHandle_t       handle,
                             hipblasOperation_t    transa,
                             hipblasOperation_t    transb,
                             int                   m,
                             int                   n,
                             int                   k,
                             const hipblasComplex* alpha,
                             const hipblasComplex* A,
                             int                   lda,
                             const hipblasComplex* B,
                             int                   ldb,
                             const hipblasComplex* beta,
                             hipblasComplex*       C,
                             int                   ldc)
try
{
    if (handle == nullptr || alpha == nullptr || A == nullptr || B == nullptr || C == nullptr || beta == nullptr ||
        m <= 0 || n <= 0 || k <= 0 || lda <= 0 || ldb <= 0 || ldc <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    // As per spec alpha can be device/host memory. In case of device memory *alpha will crash
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    auto is_dev_ptr = (queryCurrentPtrMode(handle) == HIPBLAS_POINTER_MODE_DEVICE);

    float _Complex h_alpha, h_beta;
    if (is_dev_ptr) {
        hipMemcpy(&h_alpha, alpha, sizeof(float _Complex), hipMemcpyDefault);
        hipMemcpy(&h_beta, beta, sizeof(float _Complex), hipMemcpyDefault);
    } else {
        h_alpha = *((float _Complex*)alpha);
        h_beta = *((float _Complex*)beta);
    }

    onemklCgemm(sycl_queue, convert(transa), convert(transb), m, n, k,
                h_alpha, (const float _Complex*)A, lda, (const float _Complex*)B, ldb,
                h_beta, (float _Complex*)C, ldc);
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZgemm(hipblasHandle_t             handle,
                             hipblasOperation_t          transa,
                             hipblasOperation_t          transb,
                             int                         m,
                             int                         n,
                             int                         k,
                             const hipblasDoubleComplex* alpha,
                             const hipblasDoubleComplex* A,
                             int                         lda,
                             const hipblasDoubleComplex* B,
                             int                         ldb,
                             const hipblasDoubleComplex* beta,
                             hipblasDoubleComplex*       C,
                             int                         ldc)
try
{
    if (handle == nullptr || alpha == nullptr || A == nullptr || B == nullptr || C == nullptr || beta == nullptr ||
        m <= 0 || n <= 0 || k <= 0 || lda <= 0 || ldb <= 0 || ldc <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    // As per spec alpha can be device/host memory. In case of device memory *alpha will crash
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    auto is_dev_ptr = (queryCurrentPtrMode(handle) == HIPBLAS_POINTER_MODE_DEVICE);

    double _Complex h_alpha, h_beta;
    if (is_dev_ptr) {
        hipMemcpy(&h_alpha, alpha, sizeof(double _Complex), hipMemcpyDefault);
        hipMemcpy(&h_beta, beta, sizeof(double _Complex), hipMemcpyDefault);
    } else {
        h_alpha = *((double _Complex*)alpha);
        h_beta = *((double _Complex*)beta);
    }

    onemklZgemm(sycl_queue, convert(transa), convert(transb), m, n, k,
                h_alpha, (const double _Complex*)A, lda, (const double _Complex*)B, ldb,
                h_beta, (double _Complex*)C, ldc);
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

// gemm_batched
hipblasStatus_t hipblasHgemmBatched(hipblasHandle_t          handle,
                                    hipblasOperation_t       transa,
                                    hipblasOperation_t       transb,
                                    int                      m,
                                    int                      n,
                                    int                      k,
                                    const hipblasHalf*       alpha,
                                    const hipblasHalf* const A[],
                                    int                      lda,
                                    const hipblasHalf* const B[],
                                    int                      ldb,
                                    const hipblasHalf*       beta,
                                    hipblasHalf* const       C[],
                                    int                      ldc,
                                    int                      batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasSgemmBatched(hipblasHandle_t    handle,
                                    hipblasOperation_t transa,
                                    hipblasOperation_t transb,
                                    int                m,
                                    int                n,
                                    int                k,
                                    const float*       alpha,
                                    const float* const A[],
                                    int                lda,
                                    const float* const B[],
                                    int                ldb,
                                    const float*       beta,
                                    float* const       C[],
                                    int                ldc,
                                    int                batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDgemmBatched(hipblasHandle_t     handle,
                                    hipblasOperation_t  transa,
                                    hipblasOperation_t  transb,
                                    int                 m,
                                    int                 n,
                                    int                 k,
                                    const double*       alpha,
                                    const double* const A[],
                                    int                 lda,
                                    const double* const B[],
                                    int                 ldb,
                                    const double*       beta,
                                    double* const       C[],
                                    int                 ldc,
                                    int                 batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCgemmBatched(hipblasHandle_t             handle,
                                    hipblasOperation_t          transa,
                                    hipblasOperation_t          transb,
                                    int                         m,
                                    int                         n,
                                    int                         k,
                                    const hipblasComplex*       alpha,
                                    const hipblasComplex* const A[],
                                    int                         lda,
                                    const hipblasComplex* const B[],
                                    int                         ldb,
                                    const hipblasComplex*       beta,
                                    hipblasComplex* const       C[],
                                    int                         ldc,
                                    int                         batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZgemmBatched(hipblasHandle_t                   handle,
                                    hipblasOperation_t                transa,
                                    hipblasOperation_t                transb,
                                    int                               m,
                                    int                               n,
                                    int                               k,
                                    const hipblasDoubleComplex*       alpha,
                                    const hipblasDoubleComplex* const A[],
                                    int                               lda,
                                    const hipblasDoubleComplex* const B[],
                                    int                               ldb,
                                    const hipblasDoubleComplex*       beta,
                                    hipblasDoubleComplex* const       C[],
                                    int                               ldc,
                                    int                               batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

// gemm_strided_batched
hipblasStatus_t hipblasHgemmStridedBatched(hipblasHandle_t    handle,
                                           hipblasOperation_t transa,
                                           hipblasOperation_t transb,
                                           int                m,
                                           int                n,
                                           int                k,
                                           const hipblasHalf* alpha,
                                           const hipblasHalf* A,
                                           int                lda,
                                           long long          bsa,
                                           const hipblasHalf* B,
                                           int                ldb,
                                           long long          bsb,
                                           const hipblasHalf* beta,
                                           hipblasHalf*       C,
                                           int                ldc,
                                           long long          bsc,
                                           int                batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasSgemmStridedBatched(hipblasHandle_t    handle,
                                           hipblasOperation_t transa,
                                           hipblasOperation_t transb,
                                           int                m,
                                           int                n,
                                           int                k,
                                           const float*       alpha,
                                           const float*       A,
                                           int                lda,
                                           long long          bsa,
                                           const float*       B,
                                           int                ldb,
                                           long long          bsb,
                                           const float*       beta,
                                           float*             C,
                                           int                ldc,
                                           long long          bsc,
                                           int                batchCount)
try
{
    if (handle == nullptr || alpha == nullptr || A == nullptr || B == nullptr || C == nullptr || beta == nullptr ||
        m <= 0 || n <= 0 || k <= 0 || lda <= 0 || ldb <= 0 || ldc <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    // As per spec alpha can be device/host memory. In case of device memory *alpha will crash
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    auto is_dev_ptr = (queryCurrentPtrMode(handle) == HIPBLAS_POINTER_MODE_DEVICE);

    float h_alpha, h_beta;
    if (is_dev_ptr) {
        hipMemcpy(&h_alpha, alpha, sizeof(float), hipMemcpyDefault);
        hipMemcpy(&h_beta, beta, sizeof(float), hipMemcpyDefault);
    } else {
        h_alpha = *((float*)alpha);
        h_beta = *((float*)beta);
    }

    onemklSgemm_strided(sycl_queue, convert(transa), convert(transb), m, n, k,
                h_alpha, A, lda, bsa, B, ldb, bsb, h_beta, C, ldc, bsc, batchCount);
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDgemmStridedBatched(hipblasHandle_t    handle,
                                           hipblasOperation_t transa,
                                           hipblasOperation_t transb,
                                           int                m,
                                           int                n,
                                           int                k,
                                           const double*      alpha,
                                           const double*      A,
                                           int                lda,
                                           long long          bsa,
                                           const double*      B,
                                           int                ldb,
                                           long long          bsb,
                                           const double*      beta,
                                           double*            C,
                                           int                ldc,
                                           long long          bsc,
                                           int                batchCount)
try
{
    if (handle == nullptr || alpha == nullptr || A == nullptr || B == nullptr || C == nullptr || beta == nullptr ||
        m <= 0 || n <= 0 || k <= 0 || lda <= 0 || ldb <= 0 || ldc <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    // As per spec alpha can be device/host memory. In case of device memory *alpha will crash
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    auto is_dev_ptr = (queryCurrentPtrMode(handle) == HIPBLAS_POINTER_MODE_DEVICE);

    double h_alpha, h_beta;
    if (is_dev_ptr) {
        hipMemcpy(&h_alpha, alpha, sizeof(double), hipMemcpyDefault);
        hipMemcpy(&h_beta, beta, sizeof(double), hipMemcpyDefault);
    } else {
        h_alpha = *((double*)alpha);
        h_beta = *((double*)beta);
    }

    onemklDgemm_strided(sycl_queue, convert(transa), convert(transb), m, n, k,
                h_alpha, A, lda, bsa, B, ldb, bsb, h_beta, C, ldc, bsc, batchCount);
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCgemmStridedBatched(hipblasHandle_t       handle,
                                           hipblasOperation_t    transa,
                                           hipblasOperation_t    transb,
                                           int                   m,
                                           int                   n,
                                           int                   k,
                                           const hipblasComplex* alpha,
                                           const hipblasComplex* A,
                                           int                   lda,
                                           long long             bsa,
                                           const hipblasComplex* B,
                                           int                   ldb,
                                           long long             bsb,
                                           const hipblasComplex* beta,
                                           hipblasComplex*       C,
                                           int                   ldc,
                                           long long             bsc,
                                           int                   batchCount)
try
{
    if (handle == nullptr || alpha == nullptr || A == nullptr || B == nullptr || C == nullptr || beta == nullptr ||
        m <= 0 || n <= 0 || k <= 0 || lda <= 0 || ldb <= 0 || ldc <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    // As per spec alpha can be device/host memory. In case of device memory *alpha will crash
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    auto is_dev_ptr = (queryCurrentPtrMode(handle) == HIPBLAS_POINTER_MODE_DEVICE);

    float _Complex h_alpha, h_beta;
    if (is_dev_ptr) {
        hipMemcpy(&h_alpha, alpha, sizeof(float _Complex), hipMemcpyDefault);
        hipMemcpy(&h_beta, beta, sizeof(float _Complex), hipMemcpyDefault);
    } else {
        h_alpha = *((float _Complex*)alpha);
        h_beta = *((float _Complex*)beta);
    }

    onemklCgemm_strided(sycl_queue, convert(transa), convert(transb), m, n, k,
                h_alpha, (const float _Complex*)A, lda, bsa, (const float _Complex*)B, ldb, bsb,
                h_beta, (float _Complex*)C, ldc, bsc, batchCount);
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZgemmStridedBatched(hipblasHandle_t             handle,
                                           hipblasOperation_t          transa,
                                           hipblasOperation_t          transb,
                                           int                         m,
                                           int                         n,
                                           int                         k,
                                           const hipblasDoubleComplex* alpha,
                                           const hipblasDoubleComplex* A,
                                           int                         lda,
                                           long long                   bsa,
                                           const hipblasDoubleComplex* B,
                                           int                         ldb,
                                           long long                   bsb,
                                           const hipblasDoubleComplex* beta,
                                           hipblasDoubleComplex*       C,
                                           int                         ldc,
                                           long long                   bsc,
                                           int                         batchCount)
try
{
    if (handle == nullptr || alpha == nullptr || A == nullptr || B == nullptr || C == nullptr || beta == nullptr ||
        m <= 0 || n <= 0 || k <= 0 || lda <= 0 || ldb <= 0 || ldc <= 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    // As per spec alpha can be device/host memory. In case of device memory *alpha will crash
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    auto is_dev_ptr = (queryCurrentPtrMode(handle) == HIPBLAS_POINTER_MODE_DEVICE);

    double _Complex h_alpha, h_beta;
    if (is_dev_ptr) {
        hipMemcpy(&h_alpha, alpha, sizeof(double _Complex), hipMemcpyDefault);
        hipMemcpy(&h_beta, beta, sizeof(double _Complex), hipMemcpyDefault);
    } else {
        h_alpha = *((double _Complex*)alpha);
        h_beta = *((double _Complex*)beta);
    }

    onemklZgemm_strided(sycl_queue, convert(transa), convert(transb), m, n, k,
                h_alpha, (const double _Complex*)A, lda, bsa, (const double _Complex*)B, ldb, bsb,
                h_beta, (double _Complex*)C, ldc, bsc, batchCount);
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

//gemm-ex
hipblasStatus_t hipblasGemmEx(hipblasHandle_t    handle,
                              hipblasOperation_t transa,
                              hipblasOperation_t transb,
                              int                m,
                              int                n,
                              int                k,
                              const void*        alpha,
                              const void*        A,
                              hipblasDatatype_t  a_type,
                              int                lda,
                              const void*        B,
                              hipblasDatatype_t  b_type,
                              int                ldb,
                              const void*        beta,
                              void*              C,
                              hipblasDatatype_t  c_type,
                              int                ldc,
                              hipblasDatatype_t  compute_type,
                              hipblasGemmAlgo_t  algo)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasGemmBatchedEx(hipblasHandle_t    handle,
                                     hipblasOperation_t transa,
                                     hipblasOperation_t transb,
                                     int                m,
                                     int                n,
                                     int                k,
                                     const void*        alpha,
                                     const void*        A[],
                                     hipblasDatatype_t  a_type,
                                     int                lda,
                                     const void*        B[],
                                     hipblasDatatype_t  b_type,
                                     int                ldb,
                                     const void*        beta,
                                     void*              C[],
                                     hipblasDatatype_t  c_type,
                                     int                ldc,
                                     int                batch_count,
                                     hipblasDatatype_t  compute_type,
                                     hipblasGemmAlgo_t  algo)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasGemmStridedBatchedEx(hipblasHandle_t    handle,
                                            hipblasOperation_t transa,
                                            hipblasOperation_t transb,
                                            int                m,
                                            int                n,
                                            int                k,
                                            const void*        alpha,
                                            const void*        A,
                                            hipblasDatatype_t  a_type,
                                            int                lda,
                                            hipblasStride      stride_A,
                                            const void*        B,
                                            hipblasDatatype_t  b_type,
                                            int                ldb,
                                            hipblasStride      stride_B,
                                            const void*        beta,
                                            void*              C,
                                            hipblasDatatype_t  c_type,
                                            int                ldc,
                                            hipblasStride      stride_C,
                                            int                batch_count,
                                            hipblasDatatype_t  compute_type,
                                            hipblasGemmAlgo_t  algo)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

// trsm_ex
hipblasStatus_t hipblasTrsmEx(hipblasHandle_t    handle,
                              hipblasSideMode_t  side,
                              hipblasFillMode_t  uplo,
                              hipblasOperation_t transA,
                              hipblasDiagType_t  diag,
                              int                m,
                              int                n,
                              const void*        alpha,
                              void*              A,
                              int                lda,
                              void*              B,
                              int                ldb,
                              const void*        invA,
                              int                invA_size,
                              hipblasDatatype_t  compute_type)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasTrsmBatchedEx(hipblasHandle_t    handle,
                                     hipblasSideMode_t  side,
                                     hipblasFillMode_t  uplo,
                                     hipblasOperation_t transA,
                                     hipblasDiagType_t  diag,
                                     int                m,
                                     int                n,
                                     const void*        alpha,
                                     void*              A,
                                     int                lda,
                                     void*              B,
                                     int                ldb,
                                     int                batch_count,
                                     const void*        invA,
                                     int                invA_size,
                                     hipblasDatatype_t  compute_type)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasTrsmStridedBatchedEx(hipblasHandle_t    handle,
                                            hipblasSideMode_t  side,
                                            hipblasFillMode_t  uplo,
                                            hipblasOperation_t transA,
                                            hipblasDiagType_t  diag,
                                            int                m,
                                            int                n,
                                            const void*        alpha,
                                            void*              A,
                                            int                lda,
                                            hipblasStride      stride_A,
                                            void*              B,
                                            int                ldb,
                                            hipblasStride      stride_B,
                                            int                batch_count,
                                            const void*        invA,
                                            int                invA_size,
                                            hipblasStride      stride_invA,
                                            hipblasDatatype_t  compute_type)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

// AXPY_ex
hipblasStatus_t hipblasAxpyEx(hipblasHandle_t   handle,
                              int               n,
                              const void*       alpha,
                              hipblasDatatype_t alphaType,
                              const void*       x,
                              hipblasDatatype_t xType,
                              int               incx,
                              void*             y,
                              hipblasDatatype_t yType,
                              int               incy,
                              hipblasDatatype_t executionType)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasAxpyBatchedEx(hipblasHandle_t   handle,
                                     int               n,
                                     const void*       alpha,
                                     hipblasDatatype_t alphaType,
                                     const void*       x,
                                     hipblasDatatype_t xType,
                                     int               incx,
                                     void*             y,
                                     hipblasDatatype_t yType,
                                     int               incy,
                                     int               batch_count,
                                     hipblasDatatype_t executionType)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasAxpyStridedBatchedEx(hipblasHandle_t   handle,
                                            int               n,
                                            const void*       alpha,
                                            hipblasDatatype_t alphaType,
                                            const void*       x,
                                            hipblasDatatype_t xType,
                                            int               incx,
                                            hipblasStride     stridex,
                                            void*             y,
                                            hipblasDatatype_t yType,
                                            int               incy,
                                            hipblasStride     stridey,
                                            int               batch_count,
                                            hipblasDatatype_t executionType)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

//dot-ex and dotc-ex
hipblasStatus_t hipblasDotEx(hipblasHandle_t   handle,
                             int               n,
                             const void*       x,
                             hipblasDatatype_t xType,
                             int               incx,
                             const void*       y,
                             hipblasDatatype_t yType,
                             int               incy,
                             void*             result,
                             hipblasDatatype_t resultType,
                             hipblasDatatype_t executionType)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDotcEx(hipblasHandle_t   handle,
                              int               n,
                              const void*       x,
                              hipblasDatatype_t xType,
                              int               incx,
                              const void*       y,
                              hipblasDatatype_t yType,
                              int               incy,
                              void*             result,
                              hipblasDatatype_t resultType,
                              hipblasDatatype_t executionType)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDotBatchedEx(hipblasHandle_t   handle,
                                    int               n,
                                    const void*       x,
                                    hipblasDatatype_t xType,
                                    int               incx,
                                    const void*       y,
                                    hipblasDatatype_t yType,
                                    int               incy,
                                    int               batch_count,
                                    void*             result,
                                    hipblasDatatype_t resultType,
                                    hipblasDatatype_t executionType)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDotcBatchedEx(hipblasHandle_t   handle,
                                     int               n,
                                     const void*       x,
                                     hipblasDatatype_t xType,
                                     int               incx,
                                     const void*       y,
                                     hipblasDatatype_t yType,
                                     int               incy,
                                     int               batch_count,
                                     void*             result,
                                     hipblasDatatype_t resultType,
                                     hipblasDatatype_t executionType)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDotStridedBatchedEx(hipblasHandle_t   handle,
                                           int               n,
                                           const void*       x,
                                           hipblasDatatype_t xType,
                                           int               incx,
                                           hipblasStride     stridex,
                                           const void*       y,
                                           hipblasDatatype_t yType,
                                           int               incy,
                                           hipblasStride     stridey,
                                           int               batch_count,
                                           void*             result,
                                           hipblasDatatype_t resultType,
                                           hipblasDatatype_t executionType)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDotcStridedBatchedEx(hipblasHandle_t   handle,
                                            int               n,
                                            const void*       x,
                                            hipblasDatatype_t xType,
                                            int               incx,
                                            hipblasStride     stridex,
                                            const void*       y,
                                            hipblasDatatype_t yType,
                                            int               incy,
                                            hipblasStride     stridey,
                                            int               batch_count,
                                            void*             result,
                                            hipblasDatatype_t resultType,
                                            hipblasDatatype_t executionType)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

// nrm2_ex
hipblasStatus_t hipblasNrm2Ex(hipblasHandle_t   handle,
                              int               n,
                              const void*       x,
                              hipblasDatatype_t xType,
                              int               incx,
                              void*             result,
                              hipblasDatatype_t resultType,
                              hipblasDatatype_t executionType)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasNrm2BatchedEx(hipblasHandle_t   handle,
                                     int               n,
                                     const void*       x,
                                     hipblasDatatype_t xType,
                                     int               incx,
                                     int               batch_count,
                                     void*             result,
                                     hipblasDatatype_t resultType,
                                     hipblasDatatype_t executionType)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasNrm2StridedBatchedEx(hipblasHandle_t   handle,
                                            int               n,
                                            const void*       x,
                                            hipblasDatatype_t xType,
                                            int               incx,
                                            hipblasStride     stridex,
                                            int               batch_count,
                                            void*             result,
                                            hipblasDatatype_t resultType,
                                            hipblasDatatype_t executionType)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

// rot_ex
hipblasStatus_t hipblasRotEx(hipblasHandle_t   handle,
                             int               n,
                             void*             x,
                             hipblasDatatype_t xType,
                             int               incx,
                             void*             y,
                             hipblasDatatype_t yType,
                             int               incy,
                             const void*       c,
                             const void*       s,
                             hipblasDatatype_t csType,
                             hipblasDatatype_t executionType)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasRotBatchedEx(hipblasHandle_t   handle,
                                    int               n,
                                    void*             x,
                                    hipblasDatatype_t xType,
                                    int               incx,
                                    void*             y,
                                    hipblasDatatype_t yType,
                                    int               incy,
                                    const void*       c,
                                    const void*       s,
                                    hipblasDatatype_t csType,
                                    int               batch_count,
                                    hipblasDatatype_t executionType)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasRotStridedBatchedEx(hipblasHandle_t   handle,
                                           int               n,
                                           void*             x,
                                           hipblasDatatype_t xType,
                                           int               incx,
                                           hipblasStride     stridex,
                                           void*             y,
                                           hipblasDatatype_t yType,
                                           int               incy,
                                           hipblasStride     stridey,
                                           const void*       c,
                                           const void*       s,
                                           hipblasDatatype_t csType,
                                           int               batch_count,
                                           hipblasDatatype_t executionType)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

// scal_ex
hipblasStatus_t hipblasScalEx(hipblasHandle_t   handle,
                              int               n,
                              const void*       alpha,
                              hipblasDatatype_t alphaType,
                              void*             x,
                              hipblasDatatype_t xType,
                              int               incx,
                              hipblasDatatype_t executionType)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasScalBatchedEx(hipblasHandle_t   handle,
                                     int               n,
                                     const void*       alpha,
                                     hipblasDatatype_t alphaType,
                                     void*             x,
                                     hipblasDatatype_t xType,
                                     int               incx,
                                     int               batch_count,
                                     hipblasDatatype_t executionType)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasScalStridedBatchedEx(hipblasHandle_t   handle,
                                            int               n,
                                            const void*       alpha,
                                            hipblasDatatype_t alphaType,
                                            void*             x,
                                            hipblasDatatype_t xType,
                                            int               incx,
                                            hipblasStride     stridex,
                                            int               batch_count,
                                            hipblasDatatype_t executionType)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception_to_hipblas_status();
}

