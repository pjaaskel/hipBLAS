#include "onemkl.h"
#include "sycl.hpp"

#include <oneapi/mkl.hpp>

// This is a workaround to flush MKL submissions into Level-zero queue, 
// using unspecified but guaranteed behavior of intel-sycl runtime. 
// Once SYCL standard committee approves sycl::queue::flush() we will change the macro to use the same 
#define __FORCE_MKL_FLUSH__(cmd) \
            get_native<sycl::backend::ext_oneapi_level_zero>(cmd)

// gemm

// https://spec.oneapi.io/versions/1.0-rev-1/elements/oneMKL/source/domains/blas/gemm.html

oneapi::mkl::transpose convert(onemklTranspose val) {
    switch (val) {
    case ONEMKL_TRANSPOSE_NONTRANS:
        return oneapi::mkl::transpose::nontrans;
    case ONEMKL_TRANSPOSE_TRANS:
        return oneapi::mkl::transpose::trans;
    case ONEMLK_TRANSPOSE_CONJTRANS:
        return oneapi::mkl::transpose::conjtrans;
    }
}

oneapi::mkl::uplo convert(onemklUplo val) {
    switch(val) {
        case ONEMKL_UPLO_UPPER:
            return oneapi::mkl::uplo::upper;
        case ONEMKL_UPLO_LOWER:
            return oneapi::mkl::uplo::lower;
    }
}

oneapi::mkl::side convert(onemklSideMode val) {
    switch(val) {
        case ONEMKL_SIDE_LEFT:
            return oneapi::mkl::side::left;
        case ONEMKL_SIDE_RIGHT:
            return oneapi::mkl::side::right;
    }
}

oneapi::mkl::diag convert(onemklDiag val) {
    switch(val) {
        case ONEMKL_DIAG_NONUNIT:
            return oneapi::mkl::diag::nonunit;
        case ONEMKL_DIAG_UNIT:
            return oneapi::mkl::diag::unit;
    }
}

extern "C" void onemklSdot(sycl::queue device_queue, int64_t n,
                           const float *x, int64_t incx, const float *y,
                           int64_t incy, float *result) {
    auto status = oneapi::mkl::blas::column_major::dot(device_queue, n, x,
                                                       incx, y, incy, result);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklDdot(sycl::queue device_queue, int64_t n,
                           const double *x, int64_t incx, const double *y,
                           int64_t incy, double *result) {
    auto status = oneapi::mkl::blas::column_major::dot(device_queue, n, x,
                                                       incx, y, incy, result);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklCdotc(sycl::queue device_queue, int64_t n,
                           const float _Complex *x, int64_t incx, const float _Complex *y,
                           int64_t incy, float _Complex *result) {
    auto status = oneapi::mkl::blas::column_major::dotc(device_queue, n,
                                                reinterpret_cast<const std::complex<float> *>(x), incx,
                                                reinterpret_cast<const std::complex<float> *>(y), incy,
                                                reinterpret_cast<std::complex<float> *>(result));
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklZdotc(sycl::queue device_queue, int64_t n,
                           const double _Complex *x, int64_t incx, const double _Complex *y,
                           int64_t incy, double _Complex *result) {
    auto status = oneapi::mkl::blas::column_major::dotc(device_queue, n,
                                                reinterpret_cast<const std::complex<double> *>(x), incx,
                                                reinterpret_cast<const std::complex<double> *>(y), incy,
                                                reinterpret_cast<std::complex<double> *>(result));
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklCdotu(sycl::queue device_queue, int64_t n,
                           const float _Complex *x, int64_t incx, const float _Complex *y,
                           int64_t incy, float _Complex *result) {
    auto status = oneapi::mkl::blas::column_major::dotu(device_queue, n,
                                                reinterpret_cast<const std::complex<float> *>(x), incx,
                                                reinterpret_cast<const std::complex<float> *>(y), incy,
                                                reinterpret_cast<std::complex<float> *>(result));
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklZdotu(sycl::queue device_queue, int64_t n,
                           const double _Complex *x, int64_t incx, const double _Complex *y,
                           int64_t incy, double _Complex *result) {
    auto status = oneapi::mkl::blas::column_major::dotu(device_queue, n,
                                                reinterpret_cast<const std::complex<double> *>(x), incx,
                                                reinterpret_cast<const std::complex<double> *>(y), incy,
                                                reinterpret_cast<std::complex<double> *>(result));
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklSasum(sycl::queue device_queue, int64_t n, 
                            const float *x, int64_t incx,
                            float *result) {
    auto status = oneapi::mkl::blas::column_major::asum(device_queue, n, x,
                                                        incx, result);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklDasum(sycl::queue device_queue, int64_t n,
                            const double *x, int64_t incx,
                            double *result) {
    auto status = oneapi::mkl::blas::column_major::asum(device_queue, n, x,
                                                        incx, result);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklCasum(sycl::queue device_queue, int64_t n,
                            const float _Complex *x, int64_t incx,
                            float *result) {
    auto status = oneapi::mkl::blas::column_major::asum(device_queue, n, 
                                        reinterpret_cast<const std::complex<float> *>(x),
                                        incx, result);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklZasum(sycl::queue device_queue, int64_t n,
                            const double _Complex *x, int64_t incx,
                            double *result) {
    auto status = oneapi::mkl::blas::column_major::asum(device_queue, n, 
                                        reinterpret_cast<const std::complex<double> *>(x),
                                        incx, result);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklSaxpy(sycl::queue device_queue, int64_t n, float alpha,
                            const float *x, std::int64_t incx, float *y, int64_t incy) {
    auto status = oneapi::mkl::blas::column_major::axpy(device_queue, n, alpha, x,
                                                incx, y, incy);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklDaxpy(sycl::queue device_queue, int64_t n, double alpha, 
                            const double *x, std::int64_t incx, double *y, int64_t incy) {
    auto status = oneapi::mkl::blas::column_major::axpy(device_queue, n, alpha, x,
                                                incx, y, incy);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklCaxpy(sycl::queue device_queue, int64_t n, float _Complex alpha,
                        const float _Complex *x, std::int64_t incx, float _Complex *y, int64_t incy) {
    auto status = oneapi::mkl::blas::column_major::axpy(device_queue, n, alpha,
                            reinterpret_cast<const std::complex<float> *>(x), incx,
                            reinterpret_cast<std::complex<float> *>(y), incy);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklZaxpy(sycl::queue device_queue, int64_t n, double _Complex alpha,
                        const double _Complex *x, std::int64_t incx, double _Complex *y, int64_t incy) {
    auto status = oneapi::mkl::blas::column_major::axpy(device_queue, n, alpha,
                            reinterpret_cast<const std::complex<double> *>(x), incx,
                            reinterpret_cast<std::complex<double> *>(y), incy);
    __FORCE_MKL_FLUSH__(status);
}

// Support Level-1: SCAL primitive
extern "C" void onemklDscal(sycl::queue device_queue, int64_t n, double alpha,
                            double *x, int64_t incx) {
    auto status = oneapi::mkl::blas::column_major::scal(device_queue, n, alpha,
                                                    x, incx);
    __FORCE_MKL_FLUSH__(status);

}

extern "C" void onemklSscal(sycl::queue device_queue, int64_t n, float alpha,
                            float *x, int64_t incx) {
    auto status = oneapi::mkl::blas::column_major::scal(device_queue, n, alpha,
                                                         x, incx);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklCscal(sycl::queue device_queue, int64_t n,
                            float _Complex alpha, float _Complex *x,
                            int64_t incx) {
    auto status = oneapi::mkl::blas::column_major::scal(device_queue, n,
                                        static_cast<std::complex<float> >(alpha),
                                        reinterpret_cast<std::complex<float> *>(x),incx);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklCsscal(sycl::queue device_queue, int64_t n,
                            float alpha, float _Complex *x,
                            int64_t incx) {
    auto status = oneapi::mkl::blas::column_major::scal(device_queue, n, alpha,
                                        reinterpret_cast<std::complex<float> *>(x),incx);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklZscal(sycl::queue device_queue, int64_t n,
                            double _Complex alpha, double _Complex *x,
                            int64_t incx) {
    auto status = oneapi::mkl::blas::column_major::scal(device_queue, n,
                                        static_cast<std::complex<double> >(alpha),
                                        reinterpret_cast<std::complex<double> *>(x),incx);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklZdscal(sycl::queue device_queue, int64_t n,
                            double alpha, double _Complex *x,
                            int64_t incx) {
    auto status = oneapi::mkl::blas::column_major::scal(device_queue, n, alpha,
                                        reinterpret_cast<std::complex<double> *>(x),incx);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklDnrm2(sycl::queue device_queue, int64_t n, const double *x,
                            int64_t incx, double *result) {
    auto status = oneapi::mkl::blas::column_major::nrm2(device_queue, n, x, incx, result);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklSnrm2(sycl::queue device_queue, int64_t n, const float *x,
                            int64_t incx, float *result) {
    auto status = oneapi::mkl::blas::column_major::nrm2(device_queue, n, x, incx, result);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklCnrm2(sycl::queue device_queue, int64_t n, const float _Complex *x,
                            int64_t incx, float *result) {
    auto status = oneapi::mkl::blas::column_major::nrm2(device_queue, n,
                    reinterpret_cast<const std::complex<float> *>(x), incx, result);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklZnrm2(sycl::queue device_queue, int64_t n, const double _Complex *x,
                            int64_t incx, double *result) {
    auto status = oneapi::mkl::blas::column_major::nrm2(device_queue, n,
                    reinterpret_cast<const std::complex<double> *>(x), incx, result);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklDcopy(sycl::queue device_queue, int64_t n, const double *x,
                            int64_t incx, double *y, int64_t incy) {
    auto status = oneapi::mkl::blas::column_major::copy(device_queue, n, x, incx, y, incy);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklScopy(sycl::queue device_queue, int64_t n, const float *x,
                            int64_t incx, float *y, int64_t incy) {
    auto status = oneapi::mkl::blas::column_major::copy(device_queue, n, x, incx, y, incy);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklZcopy(sycl::queue device_queue, int64_t n, const double _Complex *x,
                            int64_t incx, double _Complex *y, int64_t incy) {
    auto status = oneapi::mkl::blas::column_major::copy(device_queue, n,
        reinterpret_cast<const std::complex<double> *>(x), incx,
        reinterpret_cast<std::complex<double> *>(y), incy);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklCcopy(sycl::queue device_queue, int64_t n, const float _Complex *x,
                            int64_t incx, float _Complex *y, int64_t incy) {
    auto status = oneapi::mkl::blas::column_major::copy(device_queue, n,
        reinterpret_cast<const std::complex<float> *>(x), incx,
        reinterpret_cast<std::complex<float> *>(y), incy);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklDamax(sycl::queue device_queue, int64_t n, const double *x,
                            int64_t incx, int64_t *result){
    auto status = oneapi::mkl::blas::column_major::iamax(device_queue, n, x, incx, result);
    __FORCE_MKL_FLUSH__(status);
}
extern "C" void onemklSamax(sycl::queue device_queue, int64_t n, const float  *x,
                            int64_t incx, int64_t *result){
    auto status = oneapi::mkl::blas::column_major::iamax(device_queue, n, x, incx, result);
    __FORCE_MKL_FLUSH__(status);
}
extern "C" void onemklZamax(sycl::queue device_queue, int64_t n, const double _Complex *x,
                            int64_t incx, int64_t *result){
    auto status = oneapi::mkl::blas::column_major::iamax(device_queue, n,
                            reinterpret_cast<const std::complex<double> *>(x), incx, result);
    __FORCE_MKL_FLUSH__(status);
}
extern "C" void onemklCamax(sycl::queue device_queue, int64_t n, const float _Complex *x,
                            int64_t incx, int64_t *result){
    auto status = oneapi::mkl::blas::column_major::iamax(device_queue, n,
                            reinterpret_cast<const std::complex<float> *>(x), incx, result);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklDamin(sycl::queue device_queue, int64_t n, const double *x,
                            int64_t incx, int64_t *result){
    auto status = oneapi::mkl::blas::column_major::iamin(device_queue, n, x, incx, result);
    __FORCE_MKL_FLUSH__(status);
}
extern "C" void onemklSamin(sycl::queue device_queue, int64_t n, const float  *x,
                            int64_t incx, int64_t *result){
    auto status = oneapi::mkl::blas::column_major::iamin(device_queue, n, x, incx, result);
    __FORCE_MKL_FLUSH__(status);
}
extern "C" void onemklZamin(sycl::queue device_queue, int64_t n, const double _Complex *x,
                            int64_t incx, int64_t *result){
    auto status = oneapi::mkl::blas::column_major::iamin(device_queue, n,
                            reinterpret_cast<const std::complex<double> *>(x), incx, result);
    __FORCE_MKL_FLUSH__(status);
}
extern "C" void onemklCamin(sycl::queue device_queue, int64_t n, const float _Complex *x,
                            int64_t incx, int64_t *result){
    auto status = oneapi::mkl::blas::column_major::iamin(device_queue, n,
                            reinterpret_cast<const std::complex<float> *>(x), incx, result);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklSswap(sycl::queue device_queue, int64_t n, float *x, int64_t incx,\
                            float *y, int64_t incy){
    auto status = oneapi::mkl::blas::column_major::swap(device_queue, n, x, incx, y, incy);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklDswap(sycl::queue device_queue, int64_t n, double *x, int64_t incx,
                            double *y, int64_t incy){
    auto status = oneapi::mkl::blas::column_major::swap(device_queue, n, x, incx, y, incy);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklCswap(sycl::queue device_queue, int64_t n, float _Complex *x, int64_t incx,
                            float _Complex *y, int64_t incy){
    auto status = oneapi::mkl::blas::column_major::swap(device_queue, n,
                            reinterpret_cast<std::complex<float> *>(x), incx,
                            reinterpret_cast<std::complex<float> *>(y), incy);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklZswap(sycl::queue device_queue, int64_t n, double _Complex *x, int64_t incx,
                            double _Complex *y, int64_t incy){
    auto status = oneapi::mkl::blas::column_major::swap(device_queue, n,
                            reinterpret_cast<std::complex<double> *>(x), incx,
                            reinterpret_cast<std::complex<double> *>(y), incy);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklSrot(sycl::queue device_queue, int n, float* x, int incx, float* y, int incy,
                const float c, const float s){
    auto status = oneapi::mkl::blas::column_major::rot(device_queue, n, x, incx, y, incy, c, s);
    __FORCE_MKL_FLUSH__(status);
}
extern "C" void onemklDrot(sycl::queue device_queue, int n, double* x, int incx, double* y, int incy,
                const double c, const double s){
    auto status = oneapi::mkl::blas::column_major::rot(device_queue, n, x, incx, y, incy, c, s);
    __FORCE_MKL_FLUSH__(status);
}
extern "C" void onemklCrot(sycl::queue device_queue, int n, float _Complex* x, int incx, float _Complex* y, int incy,
                const float c, const float _Complex s){
    auto status = oneapi::mkl::blas::column_major::rot(device_queue, n,
                    reinterpret_cast<std::complex<float> *>(x), incx,
                    reinterpret_cast<std::complex<float> *>(y), incy,
                    c, static_cast<std::complex<float> >(s));
    __FORCE_MKL_FLUSH__(status);
}
extern "C" void onemklCsrot(sycl::queue device_queue, int n, float _Complex* x, int incx, float _Complex* y, int incy,
                const float c, const float s){
    auto status = oneapi::mkl::blas::column_major::rot(device_queue, n,
                    reinterpret_cast<std::complex<float> *>(x), incx,
                    reinterpret_cast<std::complex<float> *>(y), incy, c, s);
    __FORCE_MKL_FLUSH__(status);
}
extern "C" void onemklZrot(sycl::queue device_queue, int n, double _Complex* x, int incx, double _Complex* y, int incy,
                const double c, const double _Complex s){
    auto status = oneapi::mkl::blas::column_major::rot(device_queue, n,
                    reinterpret_cast<std::complex<double> *>(x), incx,
                    reinterpret_cast<std::complex<double> *>(y), incy,
                    c, static_cast<std::complex<double> >(s));
    __FORCE_MKL_FLUSH__(status);
}
extern "C" void onemklZdrot(sycl::queue device_queue, int n, double _Complex* x, int incx, double _Complex* y, int incy,
                const double c, const double s){
    auto status = oneapi::mkl::blas::column_major::rot(device_queue, n,
                    reinterpret_cast<std::complex<double> *>(x), incx,
                    reinterpret_cast<std::complex<double> *>(y), incy, c, s);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklSrotg(sycl::queue device_queue, float* a, float* b, float* c, float* s){
    auto status = oneapi::mkl::blas::column_major::rotg(device_queue, a, b, c, s);
    __FORCE_MKL_FLUSH__(status);
}
extern "C" void onemklDrotg(sycl::queue device_queue, double* a, double* b, double* c, double* s){
    auto status = oneapi::mkl::blas::column_major::rotg(device_queue, a, b, c, s);
    __FORCE_MKL_FLUSH__(status);
}
extern "C" void onemklCrotg(sycl::queue device_queue, float _Complex* a, float _Complex* b, float* c, float _Complex* s){
    auto status = oneapi::mkl::blas::column_major::rotg(device_queue,
                                                        reinterpret_cast<std::complex<float> *>(a),
                                                        reinterpret_cast<std::complex<float> *>(b), c,
                                                        reinterpret_cast<std::complex<float> *>(s));
    __FORCE_MKL_FLUSH__(status);
}
extern "C" void onemklZrotg(sycl::queue device_queue, double _Complex* a, double _Complex* b, double* c, double _Complex* s){
    auto status = oneapi::mkl::blas::column_major::rotg(device_queue,
                                                        reinterpret_cast<std::complex<double> *>(a),
                                                        reinterpret_cast<std::complex<double> *>(b), c,
                                                        reinterpret_cast<std::complex<double> *>(s));
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklSrotm(sycl::queue device_queue, int64_t n, float *x, int64_t incx,
                float *y, int64_t incy, float* param) {
    auto status = oneapi::mkl::blas::column_major::rotm(device_queue, n, x, incx, y, incy, param);
    __FORCE_MKL_FLUSH__(status);
}
extern "C" void onemklDrotm(sycl::queue device_queue, int64_t n, double *x, int64_t incx,
                double *y, int64_t incy, double* param){
    auto status = oneapi::mkl::blas::column_major::rotm(device_queue, n, x, incx, y, incy, param);
    __FORCE_MKL_FLUSH__(status);
}

// Level-2
extern "C" void onemklSgbmv(sycl::queue device_queue, onemklTranspose trans,
                            int64_t m, int64_t n, int64_t kl, int64_t ku,
                            float alpha, const float *a, int64_t lda,
                            const float *x, int64_t incx, float beta, float *y,
                            int64_t incy) {
    auto status = oneapi::mkl::blas::column_major::gbmv(device_queue,
                                convert(trans), m, n, kl, ku, alpha, a, lda, x,
                                incx, beta, y, incy);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklDgbmv(sycl::queue device_queue, onemklTranspose trans,
                            int64_t m, int64_t n, int64_t kl, int64_t ku,
                            double alpha, const double *a, int64_t lda,
                            const double *x, int64_t incx, double beta, double *y,
                            int64_t incy) {
    auto status = oneapi::mkl::blas::column_major::gbmv(device_queue, convert(trans),
                                    m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklCgbmv(sycl::queue device_queue, onemklTranspose trans,
                            int64_t m, int64_t n, int64_t kl, int64_t ku,
                            float _Complex alpha, const float _Complex *a, int64_t lda,
                            const float _Complex *x, int64_t incx, float _Complex beta,
                            float _Complex *y, int64_t incy) {
    auto status = oneapi::mkl::blas::column_major::gbmv(device_queue, convert(trans),
                                    m, n, kl, ku, static_cast<std::complex<float> >(alpha),
                                    reinterpret_cast<const std::complex<float> *>(a),
                                    lda, reinterpret_cast<const std::complex<float> *>(x),
                                    incx, static_cast<std::complex<float> >(beta),
                                    reinterpret_cast<std::complex<float> *>(y), incy);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklZgbmv(sycl::queue device_queue, onemklTranspose trans,
                            int64_t m, int64_t n, int64_t kl, int64_t ku,
                            double _Complex alpha, const double _Complex *a, int64_t lda,
                            const double _Complex *x, int64_t incx, double _Complex beta,
                            double _Complex *y, int64_t incy) {
    auto status = oneapi::mkl::blas::column_major::gbmv(device_queue, convert(trans), m,
                                        n, kl, ku, static_cast<std::complex<double> >(alpha),
                                        reinterpret_cast<const std::complex<double> *>(a),
                                        lda, reinterpret_cast<const std::complex<double> *>(x), incx,
                                        static_cast<std::complex<double> >(beta),
                                        reinterpret_cast<std::complex<double> *>(y), incy);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklSgemv(sycl::queue device_queue, onemklTranspose trans,
                            int64_t m, int64_t n, float alpha, const float *a,
                            int64_t lda, const float *x, int64_t incx, float beta,
                            float *y, int64_t incy) {
    auto status = oneapi::mkl::blas::column_major::gemv(device_queue, convert(trans),
                                            m, n, alpha, a, lda, x, incx, beta, y, incy);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklDgemv(sycl::queue device_queue, onemklTranspose trans,
                            int64_t m, int64_t n, double alpha, const double *a,
                            int64_t lda, const double *x, int64_t incx, double beta,
                            double *y, int64_t incy) {
    auto status = oneapi::mkl::blas::column_major::gemv(device_queue, convert(trans),
                                            m, n, alpha, a, lda, x, incx, beta, y, incy);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklCgemv(sycl::queue device_queue, onemklTranspose trans,
                            int64_t m, int64_t n, float _Complex alpha,
                            const float _Complex *a, int64_t lda,
                            const float _Complex *x, int64_t incx,
                            float _Complex beta, float _Complex *y,
                            int64_t incy) {
    auto status = oneapi::mkl::blas::column_major::gemv(device_queue, convert(trans), m, n,
                                            static_cast<std::complex<float> >(alpha),
                                            reinterpret_cast<const std::complex<float> *>(a), lda,
                                            reinterpret_cast<const std::complex<float> *>(x), incx,
                                            static_cast<std::complex<float> >(beta),
                                            reinterpret_cast<std::complex<float> *>(y), incy);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklZgemv(sycl::queue device_queue, onemklTranspose trans,
                            int64_t m, int64_t n, double _Complex alpha,
                            const double _Complex *a, int64_t lda,
                            const double _Complex *x, int64_t incx,
                            double _Complex beta, double _Complex *y,
                            int64_t incy) {
    auto status = oneapi::mkl::blas::column_major::gemv(device_queue, convert(trans), m, n,
                                            static_cast<std::complex<double> >(alpha),
                                            reinterpret_cast<const std::complex<double> *>(a), lda,
                                            reinterpret_cast<const std::complex<double> *>(x), incx,
                                            static_cast<std::complex<double> >(beta),
                                            reinterpret_cast<std::complex<double> *>(y), incy);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklSger(sycl::queue device_queue, int64_t m, int64_t n, float alpha,
                           const float *x, int64_t incx, const float *y, int64_t incy,
                           float *a, int64_t lda) {
    auto status = oneapi::mkl::blas::column_major::ger(device_queue, m, n, alpha, x,
                                                    incx, y, incy, a, lda);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklDger(sycl::queue device_queue, int64_t m, int64_t n, double alpha,
                           const double *x, int64_t incx, const double *y, int64_t incy,
                           double *a, int64_t lda) {
    auto status = oneapi::mkl::blas::column_major::ger(device_queue, m, n, alpha, x,
                                                    incx, y, incy, a, lda);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklCgerc(sycl::queue device_queue, int64_t m, int64_t n, float _Complex alpha,
                           const float _Complex *x, int64_t incx, const float _Complex *y, int64_t incy,
                           float _Complex *a, int64_t lda) {
    auto status = oneapi::mkl::blas::column_major::gerc(device_queue, m, n,
                                            static_cast<std::complex<float> >(alpha),
                                            reinterpret_cast<const std::complex<float> *>(x), incx,
                                            reinterpret_cast<const std::complex<float> *>(y), incy,
                                            reinterpret_cast<std::complex<float> *>(a), lda);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklCgeru(sycl::queue device_queue, int64_t m, int64_t n, float _Complex alpha,
                           const float _Complex *x, int64_t incx, const float _Complex *y, int64_t incy,
                           float _Complex *a, int64_t lda) {
    auto status = oneapi::mkl::blas::column_major::geru(device_queue, m, n,
                                            static_cast<std::complex<float> >(alpha),
                                            reinterpret_cast<const std::complex<float> *>(x), incx,
                                            reinterpret_cast<const std::complex<float> *>(y), incy,
                                            reinterpret_cast<std::complex<float> *>(a), lda);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklZgerc(sycl::queue device_queue, int64_t m, int64_t n, double _Complex alpha,
                           const double _Complex *x, int64_t incx, const double _Complex *y, int64_t incy,
                           double _Complex *a, int64_t lda) {
    auto status = oneapi::mkl::blas::column_major::gerc(device_queue, m, n,
                                          static_cast<std::complex<float> >(alpha),
                                          reinterpret_cast<const std::complex<double> *>(x), incx,
                                          reinterpret_cast<const std::complex<double> *>(y), incy,
                                          reinterpret_cast<std::complex<double> *>(a), lda);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklZgeru(sycl::queue device_queue, int64_t m, int64_t n, double _Complex alpha,
                           const double _Complex *x, int64_t incx, const double _Complex *y, int64_t incy,
                           double _Complex *a, int64_t lda) {
    auto status = oneapi::mkl::blas::column_major::geru(device_queue, m, n,
                                          static_cast<std::complex<float> >(alpha),
                                          reinterpret_cast<const std::complex<double> *>(x), incx,
                                          reinterpret_cast<const std::complex<double> *>(y), incy,
                                          reinterpret_cast<std::complex<double> *>(a), lda);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklChbmv(sycl::queue device_queue, onemklUplo uplo, int64_t n,
                            int64_t k, float _Complex alpha, const float _Complex *a,
                            int64_t lda, const float _Complex *x, int64_t incx, float _Complex beta,
                            float _Complex *y, int64_t incy) {
    auto status = oneapi::mkl::blas::column_major::hbmv(device_queue, convert(uplo), n,
                                          k, static_cast<std::complex<float> >(alpha),
                                          reinterpret_cast<const std::complex<float> *>(a),
                                          lda, reinterpret_cast<const std::complex<float> *>(x),
                                          incx, static_cast<std::complex<float> >(beta),
                                          reinterpret_cast<std::complex<float> *>(y), incy);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklZhbmv(sycl::queue device_queue, onemklUplo uplo, int64_t n,
                            int64_t k, double _Complex alpha, const double _Complex *a,
                            int64_t lda, const double _Complex *x, int64_t incx, double _Complex beta,
                            double _Complex *y, int64_t incy) {
    auto status = oneapi::mkl::blas::column_major::hbmv(device_queue, convert(uplo), n,
                                          k, static_cast<std::complex<double> >(alpha),
                                          reinterpret_cast<const std::complex<double> *>(a),
                                          lda, reinterpret_cast<const std::complex<double> *>(x),
                                          incx, static_cast<std::complex<double> >(beta),
                                          reinterpret_cast<std::complex<double> *>(y), incy);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklChemv(sycl::queue device_queue, onemklUplo uplo, int64_t n,
                            float _Complex alpha, const float _Complex *a, int64_t lda,
                            const float _Complex *x, int64_t incx, float _Complex beta,
                            float _Complex *y, int64_t incy) {
    auto status = oneapi::mkl::blas::column_major::hemv(device_queue, convert(uplo), n,
                                          static_cast<std::complex<float> >(alpha),
                                          reinterpret_cast<const std::complex<float> *>(a),
                                          lda, reinterpret_cast<const std::complex<float> *>(x), incx,
                                          static_cast<std::complex<float> >(beta),
                                          reinterpret_cast<std::complex<float> *>(y), incy);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklZhemv(sycl::queue device_queue, onemklUplo uplo, int64_t n,
                            double _Complex alpha, const double _Complex *a, int64_t lda,
                            const double _Complex *x, int64_t incx, double _Complex beta,
                            double _Complex *y, int64_t incy) {
    auto status = oneapi::mkl::blas::column_major::hemv(device_queue, convert(uplo), n,
                                          static_cast<std::complex<double> >(alpha),
                                          reinterpret_cast<const std::complex<double> *>(a),
                                          lda, reinterpret_cast<const std::complex<double> *>(x), incx,
                                          static_cast<std::complex<double> >(beta),
                                          reinterpret_cast<std::complex<double> *>(y), incy);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklCher(sycl::queue device_queue, onemklUplo uplo, int64_t n, float alpha,
                           const float _Complex *x, int64_t incx, float _Complex *a,
                           int64_t lda) {
    auto status = oneapi::mkl::blas::column_major::her(device_queue, convert(uplo), n, alpha,
                                        reinterpret_cast<const std::complex<float> *>(x), incx,
                                        reinterpret_cast<std::complex<float> *>(a), lda);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklZher(sycl::queue device_queue, onemklUplo uplo, int64_t n, double alpha,
                           const double _Complex *x, int64_t incx, double _Complex *a,
                           int64_t lda) {
    auto status = oneapi::mkl::blas::column_major::her(device_queue, convert(uplo), n, alpha,
                                        reinterpret_cast<const std::complex<double> *>(x), incx,
                                        reinterpret_cast<std::complex<double> *>(a), lda);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklCher2(sycl::queue device_queue, onemklUplo uplo, int64_t n, float _Complex alpha,
                            const float _Complex *x, int64_t incx, const float _Complex *y, int64_t incy,
                            float _Complex *a, int64_t lda) {
    auto status = oneapi::mkl::blas::column_major::her2(device_queue, convert(uplo), n,
                                          static_cast<std::complex<float> >(alpha),
                                          reinterpret_cast<const std::complex<float> *>(x), incx,
                                          reinterpret_cast<const std::complex<float> *>(y), incy,
                                          reinterpret_cast<std::complex<float> *>(a), lda);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklZher2(sycl::queue device_queue, onemklUplo uplo, int64_t n, double _Complex alpha,
                            const double _Complex *x, int64_t incx, const double _Complex *y, int64_t incy,
                            double _Complex *a, int64_t lda) {
    auto status = oneapi::mkl::blas::column_major::her2(device_queue, convert(uplo), n,
                                          static_cast<std::complex<double> >(alpha),
                                          reinterpret_cast<const std::complex<double> *>(x), incx,
                                          reinterpret_cast<const std::complex<double> *>(y), incy,
                                          reinterpret_cast<std::complex<double> *>(a), lda);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklChpmv(sycl::queue device_queue, onemklUplo uplo, int64_t n, float _Complex alpha,
                const float _Complex *a, const float _Complex *x, int64_t incx,
                float _Complex beta, float _Complex *y, int64_t incy)
{
    auto status = oneapi::mkl::blas::column_major::hpmv(device_queue, convert(uplo), n,
                                        static_cast<std::complex<float> >(alpha),
                                        reinterpret_cast<const std::complex<float> *>(a),
                                        reinterpret_cast<const std::complex<float> *>(x), incx,
                                        static_cast<std::complex<float> >(beta),
                                        reinterpret_cast<std::complex<float> *>(y), incy);
    __FORCE_MKL_FLUSH__(status);
}
extern "C" void onemklZhpmv(sycl::queue device_queue, onemklUplo uplo, int64_t n, double _Complex alpha,
                const double _Complex *a, const double _Complex *x, int64_t incx,
                double _Complex beta, double _Complex *y, int64_t incy)
{
    auto status = oneapi::mkl::blas::column_major::hpmv(device_queue, convert(uplo), n,
                                        static_cast<std::complex<double> >(alpha),
                                        reinterpret_cast<const std::complex<double> *>(a),
                                        reinterpret_cast<const std::complex<double> *>(x), incx,
                                        static_cast<std::complex<double> >(beta),
                                        reinterpret_cast<std::complex<double> *>(y), incy);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklChpr(sycl::queue device_queue, onemklUplo uplo, int64_t n, float alpha,
                const float _Complex *x, int64_t incx, float _Complex *a)
{
    auto status = oneapi::mkl::blas::column_major::hpr(device_queue, convert(uplo), n,
                                        alpha, reinterpret_cast<const std::complex<float> *>(x), incx,
                                        reinterpret_cast<std::complex<float> *>(a));
    __FORCE_MKL_FLUSH__(status);
}
extern "C" void onemklZhpr(sycl::queue device_queue, onemklUplo uplo, int64_t n, double alpha,
                const double _Complex *x, int64_t incx, double _Complex *a)
{
    auto status = oneapi::mkl::blas::column_major::hpr(device_queue, convert(uplo), n,
                                        alpha, reinterpret_cast<const std::complex<double> *>(x), incx,
                                        reinterpret_cast<std::complex<double> *>(a));
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklChpr2(sycl::queue device_queue, onemklUplo uplo, int64_t n, float _Complex alpha,
                const float _Complex *x, int64_t incx, const float _Complex *y, int64_t incy, float _Complex *a)
{
    auto status = oneapi::mkl::blas::column_major::hpr2(device_queue, convert(uplo), n,
                                        static_cast<std::complex<float> >(alpha),
                                        reinterpret_cast<const std::complex<float> *>(x), incx,
                                        reinterpret_cast<const std::complex<float> *>(y), incy,
                                        reinterpret_cast<std::complex<float> *>(a));
    __FORCE_MKL_FLUSH__(status);
}
extern "C" void onemklZhpr2(sycl::queue device_queue, onemklUplo uplo, int64_t n, double _Complex alpha,
                const double _Complex *x, int64_t incx, const double _Complex *y, int64_t incy, double _Complex *a)
{
    auto status = oneapi::mkl::blas::column_major::hpr2(device_queue, convert(uplo), n,
                                        static_cast<std::complex<double> >(alpha),
                                        reinterpret_cast<const std::complex<double> *>(x), incx,
                                        reinterpret_cast<const std::complex<double> *>(y), incy,
                                        reinterpret_cast<std::complex<double> *>(a));
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklSsbmv(sycl::queue device_queue, onemklUplo uplo, int64_t n, int64_t k,
                            float alpha, const float *a, int64_t lda, const float *x,
                            int64_t incx, float beta, float *y, int64_t incy) {
    auto status = oneapi::mkl::blas::column_major::sbmv(device_queue, convert(uplo), n, k,
                                                    alpha, a, lda, x, incx, beta, y, incy);
    __FORCE_MKL_FLUSH__(status);
}
extern "C" void onemklDsbmv(sycl::queue device_queue, onemklUplo uplo, int64_t n, int64_t k,
                            double alpha, const double *a, int64_t lda, const double *x,
                            int64_t incx, double beta, double *y, int64_t incy) {
    auto status = oneapi::mkl::blas::column_major::sbmv(device_queue, convert(uplo), n, k,
                                                    alpha, a, lda, x, incx, beta, y, incy);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklSspmv(sycl::queue device_queue, onemklUplo uplo, int64_t n,
                            float alpha, const float *a, const float *x,
                            int64_t incx, float beta, float *y, int64_t incy) {
    auto status = oneapi::mkl::blas::column_major::spmv(device_queue, convert(uplo), n,
                                                    alpha, a, x, incx, beta, y, incy);
    __FORCE_MKL_FLUSH__(status);
}
extern "C" void onemklDspmv(sycl::queue device_queue, onemklUplo uplo, int64_t n,
                            double alpha, const double *a, const double *x,
                            int64_t incx, double beta, double *y, int64_t incy) {
    auto status = oneapi::mkl::blas::column_major::spmv(device_queue, convert(uplo), n,
                                                    alpha, a, x, incx, beta, y, incy);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklSspr(sycl::queue device_queue, onemklUplo uplo, int64_t n,
                 float alpha, const float *x, int64_t incx, float *a) {
    auto status = oneapi::mkl::blas::column_major::spr(device_queue, convert(uplo), n, alpha, x, incx, a);
    __FORCE_MKL_FLUSH__(status);
}
extern "C" void onemklDspr(sycl::queue device_queue, onemklUplo uplo, int64_t n,
                 double alpha, const double *x, int64_t incx, double *a) {
    auto status = oneapi::mkl::blas::column_major::spr(device_queue, convert(uplo), n, alpha, x, incx, a);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklSspr2(sycl::queue device_queue, onemklUplo uplo, int64_t n,
                 float alpha, const float *x, int64_t incx,
                 const float *y, int64_t incy, float *a) {
    auto status = oneapi::mkl::blas::column_major::spr2(device_queue, convert(uplo), n, alpha,
                 x, incx, y, incy, a);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklDspr2(sycl::queue device_queue, onemklUplo uplo, int64_t n,
                 double alpha, const double *x, int64_t incx,
                 const double *y, int64_t incy, double *a) {
    auto status = oneapi::mkl::blas::column_major::spr2(device_queue, convert(uplo), n, alpha,
                 x, incx, y, incy, a);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklSsymv(sycl::queue device_queue, onemklUplo uplo, int64_t n, float alpha,
                            const float *a, int64_t lda, const float *x, int64_t incx, float beta,
                            float *y, int64_t incy) {
    auto status = oneapi::mkl::blas::column_major::symv(device_queue, convert(uplo), n, alpha,
                                                    a, lda, x, incx, beta, y, incy);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklDsymv(sycl::queue device_queue, onemklUplo uplo, int64_t n, double alpha,
                            const double *a, int64_t lda, const double *x, int64_t incx, double beta,
                            double *y, int64_t incy) {
    auto status = oneapi::mkl::blas::column_major::symv(device_queue, convert(uplo), n, alpha,
                                                    a, lda, x, incx, beta, y, incy);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklSsyr(sycl::queue device_queue, onemklUplo uplo, int64_t n, float alpha,
                           const float *x, int64_t incx, float *a, int64_t lda) {
    auto status = oneapi::mkl::blas::column_major::syr(device_queue, convert(uplo), n, alpha,
                                                    x, incx, a, lda);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklDsyr(sycl::queue device_queue, onemklUplo uplo, int64_t n, double alpha,
                           const double *x, int64_t incx, double *a, int64_t lda) {
    auto status = oneapi::mkl::blas::column_major::syr(device_queue, convert(uplo), n, alpha,
                                                    x, incx, a, lda);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklSsyr2(sycl::queue device_queue, onemklUplo uplo, int64_t n, float alpha,
                           const float *x, int64_t incx, const float *y, int64_t incy, float *a, int64_t lda)
{
    auto status = oneapi::mkl::blas::column_major::syr2(device_queue, convert(uplo), n, alpha, x, incx, y, incy, a, lda);
    __FORCE_MKL_FLUSH__(status);
}
extern "C" void onemklDsyr2(sycl::queue device_queue, onemklUplo uplo, int64_t n, double alpha,
                           const double *x, int64_t incx, const double *y, int64_t incy, double *a, int64_t lda)

{
    auto status = oneapi::mkl::blas::column_major::syr2(device_queue, convert(uplo), n, alpha, x, incx, y, incy, a, lda);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklStbmv(sycl::queue device_queue, onemklUplo uplo,
                            onemklTranspose trans, onemklDiag diag, int64_t n,
                            int64_t k, const float *a, int64_t lda, float *x, int64_t incx) {
    auto status = oneapi::mkl::blas::column_major::tbmv(device_queue, convert(uplo), convert(trans),
                                                        convert(diag), n, k, a, lda, x, incx);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklDtbmv(sycl::queue device_queue, onemklUplo uplo,
                            onemklTranspose trans, onemklDiag diag, int64_t n,
                            int64_t k, const double *a, int64_t lda, double *x, int64_t incx) {
    auto status = oneapi::mkl::blas::column_major::tbmv(device_queue, convert(uplo), convert(trans),
                                                    convert(diag), n, k, a, lda, x, incx);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklCtbmv(sycl::queue device_queue, onemklUplo uplo,
                            onemklTranspose trans, onemklDiag diag, int64_t n,
                            int64_t k, const float _Complex *a, int64_t lda, float _Complex *x,
                            int64_t incx) {
    auto status = oneapi::mkl::blas::column_major::tbmv(device_queue, convert(uplo), convert(trans),
                                            convert(diag), n, k, reinterpret_cast<const std::complex<float> *>(a),
                                            lda, reinterpret_cast<std::complex<float> *>(x), incx);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklZtbmv(sycl::queue device_queue, onemklUplo uplo,
                            onemklTranspose trans, onemklDiag diag, int64_t n,
                            int64_t k, const double _Complex *a, int64_t lda, double _Complex *x,
                            int64_t incx) {
    auto status = oneapi::mkl::blas::column_major::tbmv(device_queue, convert(uplo), convert(trans),
                                        convert(diag), n, k, reinterpret_cast<const std::complex<double> *>(a),
                                        lda, reinterpret_cast<std::complex<double> *>(x), incx);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklStbsv(sycl::queue device_queue, onemklUplo uplo,
                            onemklTranspose trans, onemklDiag diag, int64_t n,
                            int64_t k, const float *a, int64_t lda, float *x, int64_t incx) {
    auto status = oneapi::mkl::blas::column_major::tbsv(device_queue, convert(uplo), convert(trans),
                                                        convert(diag), n, k, a, lda, x, incx);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklDtbsv(sycl::queue device_queue, onemklUplo uplo,
                            onemklTranspose trans, onemklDiag diag, int64_t n,
                            int64_t k, const double *a, int64_t lda, double *x, int64_t incx) {
    auto status = oneapi::mkl::blas::column_major::tbsv(device_queue, convert(uplo), convert(trans),
                                                    convert(diag), n, k, a, lda, x, incx);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklCtbsv(sycl::queue device_queue, onemklUplo uplo,
                            onemklTranspose trans, onemklDiag diag, int64_t n,
                            int64_t k, const float _Complex *a, int64_t lda, float _Complex *x,
                            int64_t incx) {
    auto status = oneapi::mkl::blas::column_major::tbsv(device_queue, convert(uplo), convert(trans),
                                            convert(diag), n, k, reinterpret_cast<const std::complex<float> *>(a),
                                            lda, reinterpret_cast<std::complex<float> *>(x), incx);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklZtbsv(sycl::queue device_queue, onemklUplo uplo,
                            onemklTranspose trans, onemklDiag diag, int64_t n,
                            int64_t k, const double _Complex *a, int64_t lda, double _Complex *x,
                            int64_t incx) {
    auto status = oneapi::mkl::blas::column_major::tbsv(device_queue, convert(uplo), convert(trans),
                                        convert(diag), n, k, reinterpret_cast<const std::complex<double> *>(a),
                                        lda, reinterpret_cast<std::complex<double> *>(x), incx);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklStpmv(sycl::queue device_queue, onemklUplo uplo,
                onemklTranspose trans, onemklDiag diag, int64_t n,
                const float *a, float *x, int64_t incx) {
    auto status = oneapi::mkl::blas::column_major::tpmv(device_queue, convert(uplo), convert(trans),
                                                    convert(diag), n, a, x, incx);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklDtpmv(sycl::queue device_queue, onemklUplo uplo,
                onemklTranspose trans, onemklDiag diag, int64_t n,
                const double *a, double *x, int64_t incx) {
    auto status = oneapi::mkl::blas::column_major::tpmv(device_queue, convert(uplo), convert(trans),
                                                    convert(diag), n, a, x, incx);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklCtpmv(sycl::queue device_queue, onemklUplo uplo,
                onemklTranspose trans, onemklDiag diag, int64_t n,
                const float _Complex *a, float _Complex *x, int64_t incx) {
    auto status = oneapi::mkl::blas::column_major::tpmv(device_queue, convert(uplo), convert(trans),
                                            convert(diag), n, reinterpret_cast<const std::complex<float> *>(a),
                                            reinterpret_cast<std::complex<float> *>(x), incx);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklZtpmv(sycl::queue device_queue, onemklUplo uplo,
                onemklTranspose trans, onemklDiag diag, int64_t n,
                const double _Complex *a, double _Complex *x, int64_t incx) {
    auto status = oneapi::mkl::blas::column_major::tpmv(device_queue, convert(uplo), convert(trans),
                                        convert(diag), n, reinterpret_cast<const std::complex<double> *>(a),
                                        reinterpret_cast<std::complex<double> *>(x), incx);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklStpsv(sycl::queue device_queue, onemklUplo uplo,
                onemklTranspose trans, onemklDiag diag, int64_t m,
                const float *a, float *x, int64_t incx) {
    auto status = oneapi::mkl::blas::column_major::tpsv(device_queue, convert(uplo), convert(trans),
                                                    convert(diag), m, a, x, incx);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklDtpsv(sycl::queue device_queue, onemklUplo uplo,
                onemklTranspose trans, onemklDiag diag, int64_t m,
                const double *a, double *x, int64_t incx) {
    auto status = oneapi::mkl::blas::column_major::tpsv(device_queue, convert(uplo), convert(trans),
                                                    convert(diag), m, a, x, incx);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklCtpsv(sycl::queue device_queue, onemklUplo uplo,
                onemklTranspose trans, onemklDiag diag, int64_t m,
                const float _Complex *a, float _Complex *x, int64_t incx) {
    auto status = oneapi::mkl::blas::column_major::tpsv(device_queue, convert(uplo), convert(trans),
                                            convert(diag), m, reinterpret_cast<const std::complex<float> *>(a),
                                            reinterpret_cast<std::complex<float> *>(x), incx);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklZtpsv(sycl::queue device_queue, onemklUplo uplo,
                onemklTranspose trans, onemklDiag diag, int64_t m,
                const double _Complex *a, double _Complex *x, int64_t incx) {
    auto status = oneapi::mkl::blas::column_major::tpsv(device_queue, convert(uplo), convert(trans),
                                        convert(diag), m, reinterpret_cast<const std::complex<double> *>(a),
                                        reinterpret_cast<std::complex<double> *>(x), incx);
    __FORCE_MKL_FLUSH__(status);
}

// trmv - level2
extern "C" void onemklStrmv(sycl::queue device_queue, onemklUplo uplo, onemklTranspose trans,
                            onemklDiag diag, int64_t n, const float *a, int64_t lda, float *x,
                            int64_t incx) {
    auto status = oneapi::mkl::blas::column_major::trmv(device_queue, convert(uplo), convert(trans),
                                        convert(diag), n, a, lda, x, incx);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklDtrmv(sycl::queue device_queue, onemklUplo uplo, onemklTranspose trans,
                            onemklDiag diag, int64_t n, const double *a, int64_t lda, double *x,
                            int64_t incx) {
    auto status = oneapi::mkl::blas::column_major::trmv(device_queue, convert(uplo), convert(trans),
                                        convert(diag), n, a, lda, x, incx);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklCtrmv(sycl::queue device_queue, onemklUplo uplo, onemklTranspose trans,
                            onemklDiag diag, int64_t n, const float _Complex *a, int64_t lda, float _Complex *x,
                            int64_t incx) {
    auto status = oneapi::mkl::blas::column_major::trmv(device_queue, convert(uplo), convert(trans),
                                        convert(diag), n, reinterpret_cast<const std::complex<float> *>(a),
                                        lda, reinterpret_cast<std::complex<float> *>(x), incx);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklZtrmv(sycl::queue device_queue, onemklUplo uplo, onemklTranspose trans,
                            onemklDiag diag, int64_t n, const double _Complex *a, int64_t lda, double _Complex *x,
                            int64_t incx) {
    auto status = oneapi::mkl::blas::column_major::trmv(device_queue, convert(uplo), convert(trans),
                                        convert(diag), n, reinterpret_cast<const std::complex<double> *>(a),
                                        lda, reinterpret_cast<std::complex<double> *>(x), incx);
    __FORCE_MKL_FLUSH__(status);
}

// trsv
extern "C" void onemklStrsv(sycl::queue device_queue, onemklUplo uplo, onemklTranspose trans,
                            onemklDiag diag, int64_t n, const float *a, int64_t lda, float *x,
                            int64_t incx) {
    auto status = oneapi::mkl::blas::column_major::trsv(device_queue, convert(uplo), convert(trans),
                                          convert(diag), n, a, lda, x, incx);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklDtrsv(sycl::queue device_queue, onemklUplo uplo, onemklTranspose trans,
                            onemklDiag diag, int64_t n, const double *a, int64_t lda, double *x,
                            int64_t incx) {
    auto status = oneapi::mkl::blas::column_major::trsv(device_queue, convert(uplo), convert(trans),
                                          convert(diag), n, a, lda, x, incx);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklCtrsv(sycl::queue device_queue, onemklUplo uplo, onemklTranspose trans,
                            onemklDiag diag, int64_t n, const float  _Complex *a, int64_t lda,
                            float _Complex *x, int64_t incx) {
    auto status = oneapi::mkl::blas::column_major::trsv(device_queue, convert(uplo), convert(trans),
                                          convert(diag), n, reinterpret_cast<const std::complex<float> *>(a),
                                          lda, reinterpret_cast<std::complex<float> *>(x), incx);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklZtrsv(sycl::queue device_queue, onemklUplo uplo, onemklTranspose trans,
                            onemklDiag diag, int64_t n, const double _Complex *a, int64_t lda,
                            double _Complex *x, int64_t incx) {
    auto status = oneapi::mkl::blas::column_major::trsv(device_queue, convert(uplo), convert(trans),
                                          convert(diag), n, reinterpret_cast<const std::complex<double> *>(a),
                                          lda, reinterpret_cast<std::complex<double> *>(x), incx);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" int onemklHgemm(sycl::queue device_queue, onemklTranspose transA,
                           onemklTranspose transB, int64_t m, int64_t n,
                           int64_t k, sycl::half alpha, const sycl::half *A, int64_t lda,
                           const sycl::half *B, int64_t ldb, sycl::half beta, sycl::half *C,
                           int64_t ldc) {
    auto status = oneapi::mkl::blas::column_major::gemm(device_queue, convert(transA),
                                          convert(transB), m, n, k, alpha, A,
                                          lda, B, ldb, beta, C, ldc);
    __FORCE_MKL_FLUSH__(status);
    return 0;
}

extern "C" int onemklSgemm(sycl::queue device_queue, onemklTranspose transA,
                           onemklTranspose transB, int64_t m, int64_t n,
                           int64_t k, float alpha, const float *A, int64_t lda,
                           const float *B, int64_t ldb, float beta, float *C,
                           int64_t ldc) {
    auto status = oneapi::mkl::blas::column_major::gemm(device_queue, convert(transA),
                                          convert(transB), m, n, k, alpha, A,
                                          lda, B, ldb, beta, C, ldc);
    __FORCE_MKL_FLUSH__(status);
    return 0;
}

extern "C" int onemklDgemm(sycl::queue device_queue, onemklTranspose transA,
                           onemklTranspose transB, int64_t m, int64_t n,
                           int64_t k, double alpha, const double *A,
                           int64_t lda, const double *B, int64_t ldb,
                           double beta, double *C, int64_t ldc) {
    auto status = oneapi::mkl::blas::column_major::gemm(device_queue, convert(transA),
                                          convert(transB), m, n, k, alpha, A,
                                          lda, B, ldb, beta, C, ldc);
    __FORCE_MKL_FLUSH__(status);
    return 0;
}

extern "C" int onemklCgemm(sycl::queue device_queue, onemklTranspose transA,
                           onemklTranspose transB, int64_t m, int64_t n,
                           int64_t k, float _Complex alpha,
                           const float _Complex *A, int64_t lda,
                           const float _Complex *B, int64_t ldb,
                           float _Complex beta, float _Complex *C,
                           int64_t ldc) {
    auto status = oneapi::mkl::blas::column_major::gemm(
        device_queue, convert(transA), convert(transB), m, n, k, alpha,
        reinterpret_cast<const std::complex<float> *>(A), lda,
        reinterpret_cast<const std::complex<float> *>(B), ldb, beta,
        reinterpret_cast<std::complex<float> *>(C), ldc);
    __FORCE_MKL_FLUSH__(status);
    return 0;
}

extern "C" int onemklZgemm(sycl::queue device_queue, onemklTranspose transA,
                           onemklTranspose transB, int64_t m, int64_t n,
                           int64_t k, double _Complex alpha,
                           const double _Complex *A, int64_t lda,
                           const double _Complex *B, int64_t ldb,
                           double _Complex beta, double _Complex *C,
                           int64_t ldc) {
    auto status = oneapi::mkl::blas::column_major::gemm(
        device_queue, convert(transA), convert(transB), m, n, k, alpha,
        reinterpret_cast<const std::complex<double> *>(A), lda,
        reinterpret_cast<const std::complex<double> *>(B), ldb, beta,
        reinterpret_cast<std::complex<double> *>(C), ldc);
    __FORCE_MKL_FLUSH__(status);
    return 0;
}

extern "C" void onemklCherk(sycl::queue device_queue, onemklUplo uplo, onemklTranspose trans, int64_t n, int64_t k,
                float alpha, const float _Complex* a, int64_t lda, float beta, float _Complex* c, int64_t ldc) {
    auto status = oneapi::mkl::blas::column_major::herk(device_queue, convert(uplo), convert(trans), n, k,
                alpha, reinterpret_cast<const std::complex<float> *>(a), lda,
                beta, reinterpret_cast<std::complex<float> *>(c), ldc);
    __FORCE_MKL_FLUSH__(status);
}
extern "C" void onemklZherk(sycl::queue device_queue, onemklUplo uplo, onemklTranspose trans, int64_t n, int64_t k,
                double alpha, const double _Complex* a, int64_t lda, double beta, double _Complex* c, int64_t ldc) {
    auto status = oneapi::mkl::blas::column_major::herk(device_queue, convert(uplo), convert(trans), n, k,
                alpha, reinterpret_cast<const std::complex<double> *>(a), lda,
                beta, reinterpret_cast<std::complex<double> *>(c), ldc);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklCher2k(sycl::queue device_queue, onemklUplo uplo, onemklTranspose trans, int64_t n, int64_t k,
                float _Complex alpha, const float _Complex* a, int64_t lda, const float _Complex* b, int64_t ldb,
                float beta, float _Complex* c, int64_t ldc) {
    auto status = oneapi::mkl::blas::column_major::her2k(device_queue, convert(uplo), convert(trans), n, k,
                alpha, reinterpret_cast<const std::complex<float> *>(a), lda,
                reinterpret_cast<const std::complex<float> *>(b), ldb,
                beta, reinterpret_cast<std::complex<float> *>(c), ldc);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklZher2k(sycl::queue device_queue, onemklUplo uplo, onemklTranspose trans, int64_t n, int64_t k,
                double _Complex alpha, const double _Complex* a, int64_t lda,  const double _Complex* b, int64_t ldb,
                double beta, double _Complex* c, int64_t ldc) {
    auto status = oneapi::mkl::blas::column_major::her2k(device_queue, convert(uplo), convert(trans), n, k,
                alpha, reinterpret_cast<const std::complex<double> *>(a), lda,
                reinterpret_cast<const std::complex<double> *>(b), ldb,
                beta, reinterpret_cast<std::complex<double> *>(c), ldc);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklSsymm(sycl::queue device_queue, onemklSideMode side, onemklUplo uplo, int64_t m, int64_t n,
                float alpha, const float* a, int64_t lda, const float* b, int64_t ldb,
                float beta, float* c, int64_t ldc) {
    auto status = oneapi::mkl::blas::column_major::symm(device_queue, convert(side), convert(uplo), m, n,
                alpha, a, lda, b, ldb, beta, c, ldc);
    __FORCE_MKL_FLUSH__(status);
}
extern "C" void onemklDsymm(sycl::queue device_queue, onemklSideMode side, onemklUplo uplo, int64_t m, int64_t n,
                double alpha, const double* a, int64_t lda, const double* b, int64_t ldb,
                double beta, double* c, int64_t ldc) {
    auto status = oneapi::mkl::blas::column_major::symm(device_queue, convert(side), convert(uplo), m, n,
                alpha, a, lda, b, ldb, beta, c, ldc);
    __FORCE_MKL_FLUSH__(status);
}
extern "C" void onemklCsymm(sycl::queue device_queue, onemklSideMode side, onemklUplo uplo, int64_t m, int64_t n,
                float _Complex alpha, const float _Complex* a, int64_t lda, const float _Complex* b, int64_t ldb,
                float _Complex beta, float _Complex* c, int64_t ldc) {
    auto status = oneapi::mkl::blas::column_major::symm(device_queue, convert(side), convert(uplo), m, n,
                static_cast<std::complex<float>>(alpha), reinterpret_cast<const std::complex<float> *>(a), lda,
                reinterpret_cast<const std::complex<float> *>(b), ldb,
                static_cast<const std::complex<float>>(beta), reinterpret_cast<std::complex<float> *>(c), ldc);
    __FORCE_MKL_FLUSH__(status);
}
extern "C" void onemklZsymm(sycl::queue device_queue, onemklSideMode side, onemklUplo uplo, int64_t m, int64_t n,
                double _Complex alpha, const double _Complex* a, int64_t lda, const double _Complex* b, int64_t ldb,
                double _Complex beta, double _Complex* c, int64_t ldc) {
    auto status = oneapi::mkl::blas::column_major::symm(device_queue, convert(side), convert(uplo), m, n,
                static_cast<std::complex<double>>(alpha), reinterpret_cast<const std::complex<double> *>(a), lda,
                reinterpret_cast<const std::complex<double> *>(b), ldb,
                static_cast<std::complex<double>>(beta), reinterpret_cast<std::complex<double> *>(c), ldc);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklSsyrk(sycl::queue device_queue, onemklUplo uplo, onemklTranspose trans, int64_t n, int64_t k,
                float alpha, const float* a, int64_t lda, float beta, float* c, int64_t ldc) {
    auto status = oneapi::mkl::blas::column_major::syrk(device_queue, convert(uplo), convert(trans), n, k,
                alpha, a, lda, beta, c, ldc);
    __FORCE_MKL_FLUSH__(status);
}
extern "C" void onemklDsyrk(sycl::queue device_queue, onemklUplo uplo, onemklTranspose trans, int64_t n, int64_t k,
                double alpha, const double* a, int64_t lda, double beta, double* c, int64_t ldc) {
    auto status = oneapi::mkl::blas::column_major::syrk(device_queue, convert(uplo), convert(trans), n, k,
                alpha, a, lda, beta, c, ldc);
    __FORCE_MKL_FLUSH__(status);
}
extern "C" void onemklCsyrk(sycl::queue device_queue, onemklUplo uplo, onemklTranspose trans, int64_t n, int64_t k,
                float _Complex alpha, const float _Complex* a, int64_t lda, float _Complex beta, float _Complex* c, int64_t ldc) {
    auto status = oneapi::mkl::blas::column_major::syrk(device_queue, convert(uplo), convert(trans), n, k,
                static_cast<std::complex<float>>(alpha), reinterpret_cast<const std::complex<float> *>(a), lda,
                static_cast<std::complex<float>>(beta), reinterpret_cast<std::complex<float> *>(c), ldc);
    __FORCE_MKL_FLUSH__(status);
}
extern "C" void onemklZsyrk(sycl::queue device_queue, onemklUplo uplo, onemklTranspose trans, int64_t n, int64_t k,
                double _Complex alpha, const double _Complex* a, int64_t lda, double _Complex beta, double _Complex* c, int64_t ldc) {
    auto status = oneapi::mkl::blas::column_major::syrk(device_queue, convert(uplo), convert(trans), n, k,
                static_cast<std::complex<double>>(alpha), reinterpret_cast<const std::complex<double> *>(a), lda,
                static_cast<std::complex<double>>(beta), reinterpret_cast<std::complex<double> *>(c), ldc);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklSsyr2k(sycl::queue device_queue, onemklUplo uplo, onemklTranspose trans, int64_t n, int64_t k,
                float alpha, const float* a, int64_t lda, const float* b, int64_t ldb, float beta, float* c, int64_t ldc) {
    auto status = oneapi::mkl::blas::column_major::syr2k(device_queue, convert(uplo), convert(trans), n, k,
                alpha, a, lda, b, ldb, beta, c, ldc);
    __FORCE_MKL_FLUSH__(status);
}
extern "C" void onemklDsyr2k(sycl::queue device_queue, onemklUplo uplo, onemklTranspose trans, int64_t n, int64_t k,
                double alpha, const double* a, int64_t lda, const double* b, int64_t ldb, double beta, double* c, int64_t ldc) {
    auto status = oneapi::mkl::blas::column_major::syr2k(device_queue, convert(uplo), convert(trans), n, k,
                alpha, a, lda, b, ldb, beta, c, ldc);
    __FORCE_MKL_FLUSH__(status);
}
extern "C" void onemklCsyr2k(sycl::queue device_queue, onemklUplo uplo, onemklTranspose trans, int64_t n, int64_t k,
                float _Complex alpha, const float _Complex* a, int64_t lda, const float _Complex* b, int64_t ldb,
                float _Complex beta, float _Complex* c, int64_t ldc) {
    auto status = oneapi::mkl::blas::column_major::syr2k(device_queue, convert(uplo), convert(trans), n, k,
                static_cast<std::complex<float>>(alpha), reinterpret_cast<const std::complex<float> *>(a), lda,
                reinterpret_cast<const std::complex<float> *>(b), ldb,
                static_cast<std::complex<float>>(beta), reinterpret_cast<std::complex<float> *>(c), ldc);
    __FORCE_MKL_FLUSH__(status);
}
extern "C" void onemklZsyr2k(sycl::queue device_queue, onemklUplo uplo, onemklTranspose trans, int64_t n, int64_t k,
                double _Complex alpha, const double _Complex* a, int64_t lda, const double _Complex* b, int64_t ldb,
                double _Complex beta, double _Complex* c, int64_t ldc) {
    auto status = oneapi::mkl::blas::column_major::syr2k(device_queue, convert(uplo), convert(trans), n, k,
                static_cast<std::complex<double>>(alpha), reinterpret_cast<const std::complex<double> *>(a), lda,
                reinterpret_cast<const std::complex<double> *>(b), ldb,
                static_cast<std::complex<double>>(beta), reinterpret_cast<std::complex<double> *>(c), ldc);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklChemm(sycl::queue device_queue, onemklSideMode side, onemklUplo uplo, int64_t m, int64_t n,
                float _Complex alpha, const float _Complex* a, int64_t lda, const float _Complex* b, int64_t ldb,
                float _Complex beta, float _Complex* c, int64_t ldc) {
    auto status = oneapi::mkl::blas::column_major::hemm(device_queue, convert(side), convert(uplo), m, n,
                static_cast<std::complex<float>>(alpha), reinterpret_cast<const std::complex<float> *>(a), lda,
                reinterpret_cast<const std::complex<float> *>(b), ldb, static_cast<std::complex<float>>(beta),
                reinterpret_cast<std::complex<float> *>(c), ldc);
    __FORCE_MKL_FLUSH__(status);
}
extern "C" void onemklZhemm(sycl::queue device_queue, onemklSideMode side, onemklUplo uplo, int64_t m, int64_t n,
                double _Complex alpha, const double _Complex* a, int64_t lda, const double _Complex* b, int64_t ldb,
                double _Complex beta, double _Complex* c, int64_t ldc) {
    auto status = oneapi::mkl::blas::column_major::hemm(device_queue, convert(side), convert(uplo), m, n,
                static_cast<std::complex<double>>(alpha), reinterpret_cast<const std::complex<double> *>(a), lda,
                reinterpret_cast<const std::complex<double> *>(b), ldb, static_cast<std::complex<double>>(beta),
                reinterpret_cast<std::complex<double> *>(c), ldc);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklStrmm(sycl::queue device_queue, onemklSideMode side, onemklUplo uplo, onemklTranspose trans, onemklDiag diag,
                 int64_t m, int64_t n, float alpha, const float *a, int64_t lda, float *b, int64_t ldb) {
    auto status = oneapi::mkl::blas::column_major::trmm(device_queue, convert(side), convert(uplo), convert(trans), convert(diag),
                 m, n, alpha, a, lda, b, ldb);
    __FORCE_MKL_FLUSH__(status);
}
extern "C" void onemklDtrmm(sycl::queue device_queue, onemklSideMode side, onemklUplo uplo, onemklTranspose trans, onemklDiag diag,
                 int64_t m, int64_t n, double alpha, const double *a, int64_t lda, double *b, int64_t ldb) {
    auto status = oneapi::mkl::blas::column_major::trmm(device_queue, convert(side), convert(uplo), convert(trans), convert(diag),
                 m, n, alpha, a, lda, b, ldb);
    __FORCE_MKL_FLUSH__(status);
}
extern "C" void onemklCtrmm(sycl::queue device_queue, onemklSideMode side, onemklUplo uplo, onemklTranspose trans, onemklDiag diag,
                 int64_t m, int64_t n, float _Complex alpha, const float _Complex*a, int64_t lda, float _Complex*b, int64_t ldb) {
    auto status = oneapi::mkl::blas::column_major::trmm(device_queue, convert(side), convert(uplo), convert(trans), convert(diag),
                 m, n, static_cast<std::complex<float>>(alpha), reinterpret_cast<const std::complex<float> *>(a), lda,
                reinterpret_cast<std::complex<float> *>(b), ldb);
    __FORCE_MKL_FLUSH__(status);
}
extern "C" void onemklZtrmm(sycl::queue device_queue, onemklSideMode side, onemklUplo uplo, onemklTranspose trans, onemklDiag diag,
                 int64_t m, int64_t n, double _Complex alpha, const double _Complex *a, int64_t lda, double _Complex *b, int64_t ldb) {
    auto status = oneapi::mkl::blas::column_major::trmm(device_queue, convert(side), convert(uplo), convert(trans), convert(diag),
                 m, n, static_cast<std::complex<double>>(alpha), reinterpret_cast<const std::complex<double> *>(a), lda,
                reinterpret_cast<std::complex<double> *>(b), ldb);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklStrsm(sycl::queue device_queue, onemklSideMode side, onemklUplo uplo, onemklTranspose trans, onemklDiag diag,
                 int64_t m, int64_t n, float alpha, const float *a, int64_t lda, float *b, int64_t ldb) {
    auto status = oneapi::mkl::blas::column_major::trsm(device_queue, convert(side), convert(uplo), convert(trans), convert(diag),
                 m, n, alpha, a, lda, b, ldb);
    __FORCE_MKL_FLUSH__(status);
}
extern "C" void onemklDtrsm(sycl::queue device_queue, onemklSideMode side, onemklUplo uplo, onemklTranspose trans, onemklDiag diag,
                 int64_t m, int64_t n, double alpha, const double *a, int64_t lda, double *b, int64_t ldb) {
    auto status = oneapi::mkl::blas::column_major::trsm(device_queue, convert(side), convert(uplo), convert(trans), convert(diag),
                 m, n, alpha, a, lda, b, ldb);
    __FORCE_MKL_FLUSH__(status);
}
extern "C" void onemklCtrsm(sycl::queue device_queue, onemklSideMode side, onemklUplo uplo, onemklTranspose trans, onemklDiag diag,
                 int64_t m, int64_t n, float _Complex alpha, const float _Complex*a, int64_t lda, float _Complex*b, int64_t ldb) {
    auto status = oneapi::mkl::blas::column_major::trsm(device_queue, convert(side), convert(uplo), convert(trans), convert(diag),
                 m, n, static_cast<std::complex<float>>(alpha), reinterpret_cast<const std::complex<float> *>(a), lda,
                reinterpret_cast<std::complex<float> *>(b), ldb);
    __FORCE_MKL_FLUSH__(status);
}
extern "C" void onemklZtrsm(sycl::queue device_queue, onemklSideMode side, onemklUplo uplo, onemklTranspose trans, onemklDiag diag,
                 int64_t m, int64_t n, double _Complex alpha, const double _Complex *a, int64_t lda, double _Complex *b, int64_t ldb) {
    auto status = oneapi::mkl::blas::column_major::trsm(device_queue, convert(side), convert(uplo), convert(trans), convert(diag),
                 m, n, static_cast<std::complex<double>>(alpha), reinterpret_cast<const std::complex<double> *>(a), lda,
                reinterpret_cast<std::complex<double> *>(b), ldb);
    __FORCE_MKL_FLUSH__(status);
}

// other

// oneMKL keeps a cache of SYCL queues and tries to destroy them when unloading the library.
// that is incompatible with oneAPI.jl destroying queues before that, so expose a function
// to manually wipe the device cache when we're destroying queues.

namespace oneapi {
namespace mkl {
namespace gpu {
int clean_gpu_caches();
}
}
}

extern "C" void onemklDestroy() {
    oneapi::mkl::gpu::clean_gpu_caches();
}
