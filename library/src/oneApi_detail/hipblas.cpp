//#include <hip/hip_runtime.h>
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
    //Fix-Me : Should Sycl know hipStream_t??
    syclblas_set_stream(handle, lzHandles, nHandles, stream);
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

// ----------------------------- hipBlas APIs ------------------------------------

// Level-1 : amax (supported datatypes : float, double, complex float, complex double)
// Generic amax which can handle batched/stride/non-batched
hipblasStatus_t hipblasIsamax(hipblasHandle_t handle, int n, const float* x, int incx, int* result)
try
{
    bool is_result_dev_ptr = isDevicePointer(result);
    // Warning: result is a int* where as amax takes int64_t*
    int64_t *dev_results = (int64_t*)result;
    hipError_t hip_status;
    if (!is_result_dev_ptr)
        hip_status = hipMalloc(&dev_results, sizeof(int64_t));

    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    onemklSamax(sycl_queue, n, x, incx, dev_results);

    syclblas_queue_wait(sycl_queue); // wait until task is completed

    if (!is_result_dev_ptr) {
        int64_t results_host_memory = 0;
        hip_status = hipMemcpy(&results_host_memory, dev_results, sizeof(int64_t), hipMemcpyDefault);

        //Fix_Me : Chance of data corruption
        *result = (int)results_host_memory;

        hip_status = hipFree(&dev_results);
    }

    return HIPBLAS_STATUS_SUCCESS;
}
catch (...) {
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasIdamax(hipblasHandle_t handle, int n, const double* x, int incx, int* result)
try
{
    hipError_t hip_status;
    bool is_result_dev_ptr = isDevicePointer(result);
    // Warning: result is a int* where as amax takes int64_t*
    int64_t *dev_results = (int64_t*)result;

    if (!is_result_dev_ptr) {
        hip_status = hipMalloc(&dev_results, sizeof(int64_t));
    }

    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    onemklDamax(sycl_queue, n, x, incx, dev_results);
    syclblas_queue_wait(sycl_queue); // wait until task is completed

    if (!is_result_dev_ptr) {
        int64_t results_host_memory = 0;
        hip_status = hipMemcpy(&results_host_memory, dev_results, sizeof(int64_t), hipMemcpyDefault);

        //Fix_Me : Chance of data corruption
        *result = (int)results_host_memory;

        hip_status = hipFree(&dev_results);
    }

    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasIcamax(hipblasHandle_t handle, int n, const hipblasComplex* x, int incx, int* result)
try
{
    hipError_t hip_status;
    bool is_result_dev_ptr = isDevicePointer(result);
    // Warning: result is a int* where as amax takes int64_t*
    int64_t *dev_results = (int64_t*)result;

    if (!is_result_dev_ptr) {
        hip_status = hipMalloc(&dev_results, sizeof(int64_t));
    }

    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    onemklCamax(sycl_queue, n, (const float _Complex*)x, incx, dev_results);

    syclblas_queue_wait(sycl_queue); // wait until task is completed

    if (!is_result_dev_ptr) {
        int64_t results_host_memory = 0;
        hip_status = hipMemcpy(&results_host_memory, dev_results, sizeof(int64_t), hipMemcpyDefault);

        //Fix_Me : Chance of data corruption
        *result = (int)results_host_memory;

        hip_status = hipFree(&dev_results);
    }
    return HIPBLAS_STATUS_SUCCESS;
}
catch (...) {
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasIzamax(hipblasHandle_t handle, int n, const hipblasDoubleComplex* x, int incx, int* result)
try
{
    hipError_t hip_status;
    bool is_result_dev_ptr = isDevicePointer(result);
    // Warning: result is a int* where as amax takes int64_t*
    int64_t *dev_results = (int64_t*)result;

    if (!is_result_dev_ptr) {
        hip_status = hipMalloc(&dev_results, sizeof(int64_t));
    }

    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    onemklZamax(sycl_queue, n, (const double _Complex*)x, incx, dev_results);
    syclblas_queue_wait(sycl_queue); // wait until task is completed

    if (!is_result_dev_ptr) {
        int64_t results_host_memory = 0;
        hip_status = hipMemcpy(&results_host_memory, dev_results, sizeof(int64_t), hipMemcpyDefault);

        //Fix_Me : Chance of data corruption
        *result = (int)results_host_memory;

        hip_status = hipFree(&dev_results);
    }

    return HIPBLAS_STATUS_SUCCESS;
}
catch (...) {
    return exception_to_hipblas_status();
}

// Level-1 : amin (supported datatypes : float, double, complex float, complex double)
// Generic amin which can handle batched/stride/non-batched
hipblasStatus_t hipblasIsamin(hipblasHandle_t handle, int n, const float* x, int incx, int* result)
try
{
    hipError_t hip_status;
    bool is_result_dev_ptr = isDevicePointer(result);
    // Warning: result is a int* where as amin takes int64_t*
    int64_t *dev_results = (int64_t*)result;

    if (!is_result_dev_ptr) {
        hip_status = hipMalloc(&dev_results, sizeof(int64_t));
    }

    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    onemklSamin(sycl_queue, n, x, incx, dev_results);
    syclblas_queue_wait(sycl_queue); // wait until task is completed

    if (!is_result_dev_ptr) {
        int64_t results_host_memory = 0;
        hip_status = hipMemcpy(&results_host_memory, dev_results, sizeof(int64_t), hipMemcpyDefault);

        //Fix_Me : Chance of data corruption
        *result = (int)results_host_memory;

        hip_status = hipFree(&dev_results);
    }

    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasIdamin(hipblasHandle_t handle, int n, const double* x, int incx, int* result)
try
{
    hipError_t hip_status;
    bool is_result_dev_ptr = isDevicePointer(result);
    // Warning: result is a int* where as amin takes int64_t*
    int64_t *dev_results = (int64_t*)result;

    if (!is_result_dev_ptr) {
        hip_status = hipMalloc(&dev_results, sizeof(int64_t));
    }

    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    onemklDamin(sycl_queue, n, x, incx, dev_results);
    syclblas_queue_wait(sycl_queue); // wait until task is completed

    if (!is_result_dev_ptr) {
        int64_t results_host_memory = 0;
        hip_status = hipMemcpy(&results_host_memory, dev_results, sizeof(int64_t), hipMemcpyDefault);

        //Fix_Me : Chance of data corruption
        *result = (int)results_host_memory;

        hip_status = hipFree(&dev_results);
    }

    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasIcamin(hipblasHandle_t handle, int n, const hipblasComplex* x, int incx, int* result)
try
{
    hipError_t hip_status;
    bool is_result_dev_ptr = isDevicePointer(result);
    // Warning: result is a int* where as amin takes int64_t*
    int64_t *dev_results = (int64_t*)result;

    if (!is_result_dev_ptr) {
        hip_status = hipMalloc(&dev_results, sizeof(int64_t));
    }

    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    onemklCamin(sycl_queue, n, (const float _Complex*)x, incx, dev_results);

    syclblas_queue_wait(sycl_queue); // wait until task is completed

    if (!is_result_dev_ptr) {
        int64_t results_host_memory = 0;
        hip_status = hipMemcpy(&results_host_memory, dev_results, sizeof(int64_t), hipMemcpyDefault);

        //Fix_Me : Chance of data corruption
        *result = (int)results_host_memory;

        hip_status = hipFree(&dev_results);
    }

    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasIzamin(hipblasHandle_t handle, int n, const hipblasDoubleComplex* x, int incx, int* result)
try
{
    hipError_t hip_status;
    bool is_result_dev_ptr = isDevicePointer(result);
    // Warning: result is a int* where as amin takes int64_t*
    int64_t *dev_results = (int64_t*)result;

    if (!is_result_dev_ptr) {
        hip_status = hipMalloc(&dev_results, sizeof(int64_t));
    }

    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    onemklZamin(sycl_queue, n, (const double _Complex*)x, incx, dev_results);

    syclblas_queue_wait(sycl_queue); // wait until task is completed

    if (!is_result_dev_ptr) {
        int64_t results_host_memory = 0;
        hip_status = hipMemcpy(&results_host_memory, dev_results, sizeof(int64_t), hipMemcpyDefault);

        //Fix_Me : Chance of data corruption
        *result = (int)results_host_memory;

        hip_status = hipFree(&dev_results);
    }

    return HIPBLAS_STATUS_SUCCESS;
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

// Level-1 : axpy (supported datatypes : float, double, complex float, complex double)
// Generic axpy which can handle batched/stride/non-batched
hipblasStatus_t hipblasSaxpy(hipblasHandle_t handle, int n, const float* alpha,
                             const float* x, int incx, float* y, int incy)
try
{
    bool is_result_dev_ptr = isDevicePointer(alpha);
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    // 'alpha' can be device or host memory hence need to be copied before access
    float host_alpha_ptr = 0;
    if (is_result_dev_ptr) {
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
    bool is_result_dev_ptr = isDevicePointer(alpha);
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    // 'alpha' can be device or host memory hence need to be copied before access
    double host_alpha_ptr = 0;
    if (is_result_dev_ptr) {
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
    bool is_result_dev_ptr = isDevicePointer(alpha);
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    // 'alpha' can be device or host memory hence need to be copied before access
    float _Complex host_alpha_ptr = 0;
    if (is_result_dev_ptr) {
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
    bool is_result_dev_ptr = isDevicePointer(alpha);
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    // 'alpha' can be device or host memory hence need to be copied before access
    double _Complex host_alpha_ptr = 0;
    if (is_result_dev_ptr) {
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

// Level-1 : scal (supported datatypes : float, double, complex float, complex double)
// Generic scal which can handle batched/stride/non-batched
hipblasStatus_t
    hipblasSscal(hipblasHandle_t handle, int n, const float *alpha, float *x, int incx)
try
{
    bool is_dev_ptr = isDevicePointer(alpha);
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
    bool is_dev_ptr = isDevicePointer(alpha);
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
    bool is_dev_ptr = isDevicePointer(alpha);
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
    bool is_dev_ptr = isDevicePointer(alpha);
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
    bool is_dev_ptr = isDevicePointer(alpha);
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
    bool is_dev_ptr = isDevicePointer(alpha);
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

// Level-1 : nrm2 (supported datatypes : float, double, complex float, complex double)
// Generic nrm2 which can handle batched/stride/non-batched
hipblasStatus_t
    hipblasSnrm2(hipblasHandle_t handle, int n, const float* x, int incx, float* result)
try
{
    hipError_t status;
    bool is_result_dev_ptr = isDevicePointer(result);
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
    hipError_t status;
    bool is_result_dev_ptr = isDevicePointer(result);
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
    hipError_t status;
    bool is_result_dev_ptr = isDevicePointer(result);
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
    hipError_t status;
    bool is_result_dev_ptr = isDevicePointer(result);
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

// Level-1 : copy (supported datatypes : float, double, complex float, complex double)
// Generic copy which can handle batched/stride/non-batched
hipblasStatus_t
    hipblasScopy(hipblasHandle_t handle, int n, const float* x, int incx, float* y, int incy)
try
{
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
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    onemklZcopy(sycl_queue, n, (const double _Complex*)x, incx, (double _Complex*)y, incy);
    syclblas_queue_wait(sycl_queue);
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

// Level-1 : dot (supported datatypes : float, double, complex float, complex double)
// Generic dot which can handle batched/stride/non-batched
hipblasStatus_t hipblasSdot(hipblasHandle_t handle, int n, const float* x, int incx, const float* y, int incy, float* result)
try
{
    hipError_t hip_status;
    bool is_result_dev_ptr = isDevicePointer(result);
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
    hipError_t hip_status;
    bool is_result_dev_ptr = isDevicePointer(result);
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
    hipError_t hip_status;
    bool is_result_dev_ptr = isDevicePointer(result);
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
    hipError_t hip_status;
    bool is_result_dev_ptr = isDevicePointer(result);
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

hipblasStatus_t hipblasZdotc(hipblasHandle_t handle, int n, const hipblasDoubleComplex* x,
                             int incx, const hipblasDoubleComplex* y, int incy, hipblasDoubleComplex* result)
try
{
    hipError_t hip_status;
    bool is_result_dev_ptr = isDevicePointer(result);
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
    hipError_t hip_status;
    bool is_result_dev_ptr = isDevicePointer(result);
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

// Level-1 : swap (supported datatypes : float, double, complex float, complex double)
// Generic swap which can handle batched/stride/non-batched
hipblasStatus_t hipblasSswap(hipblasHandle_t handle, int n, float* x, int incx, float* y, int incy)
try
{
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
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    onemklZswap(sycl_queue, n, (double _Complex*)x, incx, (double _Complex*)y, incy);
    syclblas_queue_wait(sycl_queue);
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

// Level-1 : rot (supported datatypes : float, double, complex float, complex double)
// Generic rot which can handle batched/stride/non-batched
hipblasStatus_t hipblasSrot(hipblasHandle_t handle,int n, float* x,int incx,
                                           float* y, int incy,const float* c, const float* s)
try
{
    hipError_t hip_status;
    bool is_c_dev_ptr = isDevicePointer(c);
    bool is_s_dev_ptr = isDevicePointer(s);
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    float h_c, h_s;
    if (is_c_dev_ptr) {
        hip_status = hipMemcpy(&h_c, c, sizeof(float), hipMemcpyDefault);
    } else {
        h_c = *c;
    }
    if (is_s_dev_ptr) {
        hip_status = hipMemcpy(&h_s, s, sizeof(float), hipMemcpyDefault);
    } else {
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
    hipError_t hip_status;
    bool is_c_dev_ptr = isDevicePointer(c);
    bool is_s_dev_ptr = isDevicePointer(s);
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    double h_c, h_s;
    if (is_c_dev_ptr) {
        hip_status = hipMemcpy(&h_c, c, sizeof(double), hipMemcpyDefault);
    } else {
        h_c = *c;
    }
    if (is_s_dev_ptr) {
        hip_status = hipMemcpy(&h_s, s, sizeof(double), hipMemcpyDefault);
    } else {
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
    hipError_t hip_status;
    bool is_c_dev_ptr = isDevicePointer(c);
    bool is_s_dev_ptr = isDevicePointer(s);
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    float h_c;
    if (is_c_dev_ptr) {
        hip_status = hipMemcpy(&h_c, c, sizeof(float), hipMemcpyDefault);
    } else {
        h_c = *c;
    }
    float _Complex h_s;
    if (is_s_dev_ptr) {
        hip_status = hipMemcpy(&h_s, s, sizeof(float _Complex), hipMemcpyDefault);
    } else {
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
    hipError_t hip_status;
    bool is_c_dev_ptr = isDevicePointer(c);
    bool is_s_dev_ptr = isDevicePointer(s);
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    float h_c, h_s;
    if (is_c_dev_ptr) {
        hip_status = hipMemcpy(&h_c, c, sizeof(float), hipMemcpyDefault);
    } else {
        h_c = *c;
    }
    if (is_s_dev_ptr) {
        hip_status = hipMemcpy(&h_s, s, sizeof(float), hipMemcpyDefault);
    } else {
        h_s = *s;
    }
    // Fix-me : assuming c and s are host readable memory else we need to copy it to host memory before read
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
    hipError_t hip_status;
    bool is_c_dev_ptr = isDevicePointer(c);
    bool is_s_dev_ptr = isDevicePointer(s);
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    double h_c;
    if (is_c_dev_ptr) {
        hip_status = hipMemcpy(&h_c, c, sizeof(double), hipMemcpyDefault);
    } else {
        h_c = *c;
    }
    double _Complex h_s;
    if (is_s_dev_ptr) {
        hip_status = hipMemcpy(&h_s, s, sizeof(double _Complex), hipMemcpyDefault);
    } else {
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
    hipError_t hip_status;
    bool is_c_dev_ptr = isDevicePointer(c);
    bool is_s_dev_ptr = isDevicePointer(s);
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    double h_c, h_s;
    if (is_c_dev_ptr) {
        hip_status = hipMemcpy(&h_c, c, sizeof(double), hipMemcpyDefault);
    } else {
        h_c = *c;
    }
    if (is_s_dev_ptr) {
        hip_status = hipMemcpy(&h_s, s, sizeof(double), hipMemcpyDefault);
    } else {
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

// Level-1 : rotg (supported datatypes : float, double, complex float, complex double)
// Generic rotg which can handle batched/stride/non-batched
hipblasStatus_t hipblasSrotg(hipblasHandle_t handle, float* a, float* b, float* c, float* s)
try
{
    bool is_a_dev_ptr = isDevicePointer(a);
    bool is_b_dev_ptr = isDevicePointer(b);
    bool is_c_dev_ptr = isDevicePointer(c);
    bool is_s_dev_ptr = isDevicePointer(s);
    // FixMe: oneAPI supports only device pointers
    if (!is_a_dev_ptr || !is_b_dev_ptr || !is_c_dev_ptr || !is_s_dev_ptr) {
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
    bool is_a_dev_ptr = isDevicePointer(a);
    bool is_b_dev_ptr = isDevicePointer(b);
    bool is_c_dev_ptr = isDevicePointer(c);
    bool is_s_dev_ptr = isDevicePointer(s);
    // FixMe: oneAPI supports only device pointers
    if (!is_a_dev_ptr || !is_b_dev_ptr || !is_c_dev_ptr || !is_s_dev_ptr) {
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
    bool is_a_dev_ptr = isDevicePointer(a);
    bool is_b_dev_ptr = isDevicePointer(b);
    bool is_c_dev_ptr = isDevicePointer(c);
    bool is_s_dev_ptr = isDevicePointer(s);
    // FixMe: oneAPI supports only device pointers
    if (!is_a_dev_ptr || !is_b_dev_ptr || !is_c_dev_ptr || !is_s_dev_ptr) {
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
    bool is_a_dev_ptr = isDevicePointer(a);
    bool is_b_dev_ptr = isDevicePointer(b);
    bool is_c_dev_ptr = isDevicePointer(c);
    bool is_s_dev_ptr = isDevicePointer(s);
    // FixMe: oneAPI supports only device pointers
    if (!is_a_dev_ptr || !is_b_dev_ptr || !is_c_dev_ptr || !is_s_dev_ptr) {
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

hipblasStatus_t hipblasSrotmg(hipblasHandle_t handle, int n, float* x, int incx, float* y, int incy, const float* param)
try
{
    hipError_t hipStatus;
    bool is_param_dev_ptr = isDevicePointer(param);
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    float* dev_param = (float*) param;
    if (!is_param_dev_ptr) {
        hipStatus = hipMalloc(&dev_param, sizeof(float)*5);
        hipStatus = hipMemcpy(dev_param, param, sizeof(float)*5, hipMemcpyHostToDevice);
    }

    onemklSrotm(sycl_queue, n, x, incx, y, incy, dev_param);

    if (!is_param_dev_ptr) {
        syclblas_queue_wait(sycl_queue);
        hipFree(dev_param);
    }
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDrotmg(hipblasHandle_t handle, int n, double* x, int incx, double* y, int incy, const double* param)
try
{
    hipError_t hipStatus;
    bool is_param_dev_ptr = isDevicePointer(param);
    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    double* dev_param = (double*)param;
    if (!is_param_dev_ptr) {
        hipStatus = hipMalloc(&dev_param, sizeof(double)*5);
        hipStatus = hipMemcpy(dev_param, param, sizeof(double)*5, hipMemcpyHostToDevice);
    }

    onemklDrotm(sycl_queue, n, x, incx, y, incy, dev_param);

    if (!is_param_dev_ptr) {
        syclblas_queue_wait(sycl_queue);
        hipFree(dev_param);
    }
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}