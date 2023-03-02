
#pragma once

#include <stddef.h>
#include <stdint.h>
#include <sycl/sycl.hpp>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct syclDevice_st *syclDevice_t;
typedef struct syclPlatform_st *syclPlatform_t;
typedef struct syclContext_st *syclContext_t;
typedef struct syclEvent_st *syclEvent_t;
  //  typedef struct syclQueue_t sycl::queue;
  //typedef struct syclQueue_st *sycl::queue;

typedef enum {
    ONEMKL_TRANSPOSE_NONTRANS,
    ONEMKL_TRANSPOSE_TRANS,
    ONEMLK_TRANSPOSE_CONJTRANS
} onemklTranspose;

typedef enum {
    ONEMKL_UPLO_UPPER,
    ONEMKL_UPLO_LOWER
} onemklUplo;

typedef enum {
    ONEMKL_SIDE_LEFT,
    ONEMKL_SIDE_RIGHT,
    ONEMKL_SIDE_BOTH
} onemklSideMode;

typedef enum {
    ONEMKL_DIAG_NONUNIT,
    ONEMKL_DIAG_UNIT
 } onemklDiag;

  typedef enum {
    ONEMKL_JOB_NOVEC,
    ONEMKL_JOB_VEC
 } onemklJob;

void onemklSasum(sycl::queue device_queue, int64_t n,
                const float *x, int64_t incx, float *result);
void onemklDasum(sycl::queue device_queue, int64_t n,
                const double *x, int64_t incx, double *result);
void onemklCasum(sycl::queue device_queue, int64_t n,
                const float _Complex *x, int64_t incx, float *result);
void onemklZasum(sycl::queue device_queue, int64_t n,
                const double _Complex *x, int64_t incx, double *result);

void onemklSaxpy(sycl::queue device_queue, int64_t n, float alpha, const float *x,
                int64_t incx, float *y, int64_t incy);
void onemklDaxpy(sycl::queue device_queue, int64_t n, double alpha, const double *x,
                int64_t incx, double *y, int64_t incy);
void onemklCaxpy(sycl::queue device_queue, int64_t n, float _Complex alpha,
                const float _Complex *x, int64_t incx, float _Complex *y, int64_t incy);
void onemklZaxpy(sycl::queue device_queue, int64_t n, double _Complex alpha,
                const double _Complex *x, int64_t incx, double _Complex *y, int64_t incy);

// Level-1: scal oneMKL
void onemklDscal(sycl::queue device_queue, int64_t n, double alpha,
                double *x, int64_t incx);
void onemklSscal(sycl::queue device_queue, int64_t n, float alpha,
                float *x, int64_t incx);
void onemklCscal(sycl::queue device_queue, int64_t n, float _Complex alpha,
                float _Complex *x, int64_t incx);
void onemklCsscal(sycl::queue device_queue, int64_t n, float alpha,
                float _Complex *x, int64_t incx);
void onemklZscal(sycl::queue device_queue, int64_t n, double _Complex alpha,
                double _Complex *x, int64_t incx);
void onemklZdscal(sycl::queue device_queue, int64_t n, double alpha,
                double _Complex *x, int64_t incx);

// Supported Level-1: Nrm2
void onemklDnrm2(sycl::queue device_queue, int64_t n, const double *x,
                 int64_t incx, double *result);
void onemklSnrm2(sycl::queue device_queue, int64_t n, const float *x,
                 int64_t incx, float *result);
void onemklCnrm2(sycl::queue device_queue, int64_t n, const float _Complex *x,
                 int64_t incx, float *result);
void onemklZnrm2(sycl::queue device_queue, int64_t n, const double _Complex *x,
                 int64_t incx, double *result);

void onemklSdot(sycl::queue device_queue, int64_t n, const float *x,
                int64_t incx, const float *y, int64_t incy, float *result);
void onemklDdot(sycl::queue device_queue, int64_t n, const double *x,
                int64_t incx, const double *y, int64_t incy, double *result);
void onemklCdotc(sycl::queue device_queue, int64_t n, const float _Complex *x,
                int64_t incx, const float _Complex *y, int64_t incy,
                float _Complex *result);
void onemklZdotc(sycl::queue device_queue, int64_t n, const double _Complex *x,
                int64_t incx, const double _Complex *y, int64_t incy,
                double _Complex *result);
void onemklCdotu(sycl::queue device_queue, int64_t n, const float _Complex *x,
                int64_t incx, const float _Complex *y, int64_t incy,
                float _Complex *result);
void onemklZdotu(sycl::queue device_queue, int64_t n, const double _Complex *x,
                int64_t incx, const double _Complex *y, int64_t incy,
                double _Complex *result);

void onemklDcopy(sycl::queue device_queue, int64_t n, const double *x,
                 int64_t incx, double *y, int64_t incy);
void onemklScopy(sycl::queue device_queue, int64_t n, const float *x,
                 int64_t incx, float *y, int64_t incy);
void onemklZcopy(sycl::queue device_queue, int64_t n, const double _Complex *x,
                 int64_t incx, double _Complex *y, int64_t incy);
void onemklCcopy(sycl::queue device_queue, int64_t n, const float _Complex *x,
                 int64_t incx, float _Complex *y, int64_t incy);

void onemklDamax(sycl::queue device_queue, int64_t n, const double *x, int64_t incx,
                 int64_t *result);
void onemklSamax(sycl::queue device_queue, int64_t n, const float  *x, int64_t incx,
                 int64_t *result);
void onemklZamax(sycl::queue device_queue, int64_t n, const double _Complex *x, int64_t incx,
                 int64_t *result);
void onemklCamax(sycl::queue device_queue, int64_t n, const float _Complex *x, int64_t incx,
                 int64_t *result);

void onemklDamin(sycl::queue device_queue, int64_t n, const double *x, int64_t incx,
                 int64_t *result);
void onemklSamin(sycl::queue device_queue, int64_t n, const float  *x, int64_t incx,
                 int64_t *result);
void onemklZamin(sycl::queue device_queue, int64_t n, const double _Complex *x, int64_t incx,
                 int64_t *result);
void onemklCamin(sycl::queue device_queue, int64_t n, const float _Complex *x, int64_t incx,
                 int64_t *result);

void onemklSswap(sycl::queue device_queue, int64_t n, float *x, int64_t incx,
                float *y, int64_t incy);
void onemklDswap(sycl::queue device_queue, int64_t n, double *x, int64_t incx,
                double *y, int64_t incy);
void onemklCswap(sycl::queue device_queue, int64_t n, float _Complex *x, int64_t incx,
                float _Complex *y, int64_t incy);
void onemklZswap(sycl::queue device_queue, int64_t n, double _Complex *x, int64_t incx,
                double _Complex *y, int64_t incy);

void onemklSrot(sycl::queue device_queue, int n, float* x, int incx, float* y, int incy,
                const float c, const float s);
void onemklDrot(sycl::queue device_queue, int n, double* x, int incx, double* y, int incy,
                const double c, const double s);
void onemklCrot(sycl::queue device_queue, int n, float _Complex* x, int incx, float _Complex* y, int incy,
                const float c, const float _Complex s);
void onemklCsrot(sycl::queue device_queue, int n, float _Complex* x, int incx, float _Complex* y, int incy,
                const float c, const float s);
void onemklZrot(sycl::queue device_queue, int n, double _Complex* x, int incx, double _Complex* y, int incy,
                const double c, const double _Complex s);
void onemklZdrot(sycl::queue device_queue, int n, double _Complex* x, int incx, double _Complex* y, int incy,
                const double c, const double s);

void onemklSrotg(sycl::queue device_queue, float* a, float* b, float* c, float* s);
void onemklDrotg(sycl::queue device_queue, double* a, double* b, double* c, double* s);
void onemklCrotg(sycl::queue device_queue, float _Complex* a, float _Complex* b, float* c, float _Complex* s);
void onemklZrotg(sycl::queue device_queue, double _Complex* a, double _Complex* b, double* c, double _Complex* s);

void onemklSrotm(sycl::queue device_queue, int64_t n, float *x, int64_t incx,
                float *y, int64_t incy, float* param);
void onemklDrotm(sycl::queue device_queue, int64_t n, double *x, int64_t incx,
                double *y, int64_t incy, double* param);

// Level-2
void onemklSgbmv(sycl::queue device_queue, onemklTranspose trans, int64_t m,
                int64_t n, int64_t kl, int64_t ku, float alpha, const float *a,
                int64_t lda, const float *x, int64_t incx, float beta, float *y,
                int64_t incy);
void onemklDgbmv(sycl::queue device_queue, onemklTranspose trans, int64_t m,
                int64_t n, int64_t kl, int64_t ku, double alpha, const double *a,
                int64_t lda, const double *x, int64_t incx, double beta, double *y,
                int64_t incy);
void onemklCgbmv(sycl::queue device_queue, onemklTranspose trans, int64_t m,
                int64_t n, int64_t kl, int64_t ku, float _Complex alpha, const float
                _Complex *a, int64_t lda, const float _Complex *x, int64_t incx,
                float _Complex beta, float _Complex *y, int64_t incy);
void onemklZgbmv(sycl::queue device_queue, onemklTranspose trans, int64_t m,
                int64_t n, int64_t kl, int64_t ku, double _Complex alpha,
                const double _Complex *a, int64_t lda, const double _Complex *x,
                int64_t incx, double _Complex beta, double _Complex *y, int64_t incy);

void onemklSgemv(sycl::queue device_queue, onemklTranspose trans, int64_t m,
                 int64_t n, float alpha, const float *a, int64_t lda,
                 const float *x, int64_t incx, float beta, float *y, int64_t incy);
void onemklDgemv(sycl::queue device_queue, onemklTranspose trans, int64_t m,
                 int64_t n, double alpha, const double *a, int64_t lda,
                 const double *x, int64_t incx, double beta, double *y, int64_t incy);
void onemklCgemv(sycl::queue device_queue, onemklTranspose trans, int64_t m,
                 int64_t n, float _Complex alpha, const float _Complex *a, int64_t lda,
                 const float _Complex *x, int64_t incx, float _Complex beta,
                 float _Complex *y, int64_t incy);
void onemklZgemv(sycl::queue device_queue, onemklTranspose trans, int64_t m,
                 int64_t n, double _Complex alpha, const double _Complex *a, int64_t lda,
                 const double _Complex *x, int64_t incx, double _Complex beta,
                 double _Complex *y, int64_t incy);

void onemklSger(sycl::queue device_queue, int64_t m, int64_t n, float alpha,
                const float *x, int64_t incx, const float *y, int64_t incy,
                float *a, int64_t lda);
void onemklDger(sycl::queue device_queue, int64_t m, int64_t n, double alpha,
                const double *x, int64_t incx, const double *y, int64_t incy,
                double *a, int64_t lda);
void onemklCgerc(sycl::queue device_queue, int64_t m, int64_t n, float _Complex alpha,
                const float _Complex *x, int64_t incx, const float _Complex *y, int64_t incy,
                float _Complex *a, int64_t lda);
void onemklCgeru(sycl::queue device_queue, int64_t m, int64_t n, float _Complex alpha,
                const float _Complex *x, int64_t incx, const float _Complex *y, int64_t incy,
                float _Complex *a, int64_t lda);
void onemklZgerc(sycl::queue device_queue, int64_t m, int64_t n, double _Complex alpha,
                const double _Complex *x, int64_t incx, const double _Complex *y, int64_t incy,
                double _Complex *a, int64_t lda);
void onemklZgeru(sycl::queue device_queue, int64_t m, int64_t n, double _Complex alpha,
                const double _Complex *x, int64_t incx, const double _Complex *y, int64_t incy,
                double _Complex *a, int64_t lda);

void onemklChbmv(sycl::queue device_queue, onemklUplo uplo, int64_t n,
                int64_t k, float _Complex alpha, const float _Complex *a,
                int64_t lda, const float _Complex *x, int64_t incx, float _Complex beta,
                float _Complex *y, int64_t incy);
void onemklZhbmv(sycl::queue device_queue, onemklUplo uplo, int64_t n,
                int64_t k, double _Complex alpha, const double _Complex *a,
                int64_t lda, const double _Complex *x, int64_t incx, double _Complex beta,
                double _Complex *y, int64_t incy);

void onemklChemv(sycl::queue device_queue, onemklUplo uplo, int64_t n,
                float _Complex alpha, const float _Complex *a, int64_t lda,
                const float _Complex *x, int64_t incx, float _Complex beta,
                float _Complex *y, int64_t incy);
void onemklZhemv(sycl::queue device_queue, onemklUplo uplo, int64_t n,
                double _Complex alpha, const double _Complex *a, int64_t lda,
                const double _Complex *x, int64_t incx, double _Complex beta,
                double _Complex *y, int64_t incy);

void onemklCher(sycl::queue device_queue, onemklUplo uplo, int64_t n, float alpha,
                const float _Complex *x, int64_t incx, float _Complex *a,
                int64_t lda);
void onemklZher(sycl::queue device_queue, onemklUplo uplo, int64_t n, double alpha,
                const double _Complex *x, int64_t incx, double _Complex *a,
                int64_t lda);

void onemklCher2(sycl::queue device_queue, onemklUplo uplo, int64_t n, float _Complex alpha,
                const float _Complex *x, int64_t incx, const float _Complex *y, int64_t incy,
                float _Complex *a, int64_t lda);
void onemklZher2(sycl::queue device_queue, onemklUplo uplo, int64_t n, double _Complex alpha,
                const double _Complex *x, int64_t incx, const double _Complex *y, int64_t incy,
                double _Complex *a, int64_t lda);

void onemklChpmv(sycl::queue device_queue, onemklUplo uplo, int64_t n, float _Complex alpha,
                const float _Complex *a, const float _Complex *x, int64_t incx,
                float _Complex beta, float _Complex *y, int64_t incy);
void onemklZhpmv(sycl::queue device_queue, onemklUplo uplo, int64_t n, double _Complex alpha,
                const double _Complex *a, const double _Complex *x, int64_t incx,
                double _Complex beta, double _Complex *y, int64_t incy);

void onemklChpr(sycl::queue device_queue, onemklUplo uplo, int64_t n, float alpha,
                const float _Complex *x, int64_t incx, float _Complex *a);
void onemklZhpr(sycl::queue device_queue, onemklUplo uplo, int64_t n, double alpha,
                const double _Complex *x, int64_t incx, double _Complex *a);

void onemklChpr2(sycl::queue device_queue, onemklUplo uplo, int64_t n, float _Complex alpha,
                const float _Complex *x, int64_t incx, const float _Complex *y, int64_t incy, float _Complex *a);
void onemklZhpr2(sycl::queue device_queue, onemklUplo uplo, int64_t n, double _Complex alpha,
                const double _Complex *x, int64_t incx, const double _Complex *y, int64_t incy, double _Complex *a);

void onemklSsbmv(sycl::queue device_queue, onemklUplo uplo, int64_t n, int64_t k,
                 float alpha, const float *a, int64_t lda, const float *x,
                 int64_t incx, float beta, float *y, int64_t incy);
void onemklDsbmv(sycl::queue device_queue, onemklUplo uplo, int64_t n, int64_t k,
                 double alpha, const double *a, int64_t lda, const double *x,
                 int64_t incx, double beta, double *y, int64_t incy);

void onemklSspmv(sycl::queue device_queue, onemklUplo uplo, int64_t n,
                 float alpha, const float *a, const float *x,
                 int64_t incx, float beta, float *y, int64_t incy);
void onemklDspmv(sycl::queue device_queue, onemklUplo uplo, int64_t n,
                 double alpha, const double *a, const double *x,
                 int64_t incx, double beta, double *y, int64_t incy);

void onemklSspr(sycl::queue device_queue, onemklUplo uplo, int64_t n,
                 float alpha, const float *x, int64_t incx, float *a);
void onemklDspr(sycl::queue device_queue, onemklUplo uplo, int64_t n,
                 double alpha, const double *x, int64_t incx, double *a);

void onemklSspr2(sycl::queue device_queue, onemklUplo uplo, int64_t n,
                 float alpha, const float *x, int64_t incx,
                 const float *y, int64_t incy, float *a);
void onemklDspr2(sycl::queue device_queue, onemklUplo uplo, int64_t n,
                 double alpha, const double *x, int64_t incx,
                 const double *y, int64_t incy, double *a);

void onemklSsymv(sycl::queue device_queue, onemklUplo uplo, int64_t n, float alpha,
                 const float *a, int64_t lda, const float *x, int64_t incx, float beta,
                 float *y, int64_t incy);
void onemklDsymv(sycl::queue device_queue, onemklUplo uplo, int64_t n,
                 double alpha, const double *a, int64_t lda, const double *x,
                 int64_t incx, double beta, double *y, int64_t incy);

void onemklSsyr(sycl::queue device_queue, onemklUplo uplo, int64_t n, float alpha,
                           const float *x, int64_t incx, float *a, int64_t lda);
void onemklDsyr(sycl::queue device_queue, onemklUplo uplo, int64_t n, double alpha,
                           const double *x, int64_t incx, double *a, int64_t lda);

void onemklSsyr2(sycl::queue device_queue, onemklUplo uplo, int64_t n, float alpha,
                           const float *x, int64_t incx, const float *y, int64_t incy, float *a, int64_t lda);
void onemklDsyr2(sycl::queue device_queue, onemklUplo uplo, int64_t n, double alpha,
                           const double *x, int64_t incx, const double *y, int64_t incy, double *a, int64_t lda);

void onemklStbmv(sycl::queue device_queue, onemklUplo uplo,
                onemklTranspose trans, onemklDiag diag, int64_t n,
                int64_t k, const float *a, int64_t lda, float *x, int64_t incx);

void onemklDtbmv(sycl::queue device_queue, onemklUplo uplo,
                onemklTranspose trans, onemklDiag diag, int64_t n,
                int64_t k, const double *a, int64_t lda, double *x, int64_t incx);

void onemklCtbmv(sycl::queue device_queue, onemklUplo uplo,
                onemklTranspose trans, onemklDiag diag, int64_t n,
                int64_t k, const float _Complex *a, int64_t lda, float _Complex *x,
                int64_t incx);

void onemklZtbmv(sycl::queue device_queue, onemklUplo uplo,
                onemklTranspose trans, onemklDiag diag, int64_t n,
                int64_t k, const double _Complex *a, int64_t lda, double _Complex *x,
                int64_t incx);

void onemklStbsv(sycl::queue device_queue, onemklUplo uplo,
                onemklTranspose trans, onemklDiag diag, int64_t n,
                int64_t k, const float *a, int64_t lda, float *x, int64_t incx);

void onemklDtbsv(sycl::queue device_queue, onemklUplo uplo,
                onemklTranspose trans, onemklDiag diag, int64_t n,
                int64_t k, const double *a, int64_t lda, double *x, int64_t incx);

void onemklCtbsv(sycl::queue device_queue, onemklUplo uplo,
                onemklTranspose trans, onemklDiag diag, int64_t n,
                int64_t k, const float _Complex *a, int64_t lda, float _Complex *x,
                int64_t incx);

void onemklZtbsv(sycl::queue device_queue, onemklUplo uplo,
                onemklTranspose trans, onemklDiag diag, int64_t n,
                int64_t k, const double _Complex *a, int64_t lda, double _Complex *x,
                int64_t incx);

void onemklStpmv(sycl::queue device_queue, onemklUplo uplo,
                onemklTranspose trans, onemklDiag diag, int64_t n,
                const float *a, float *x, int64_t incx);

void onemklDtpmv(sycl::queue device_queue, onemklUplo uplo,
                onemklTranspose trans, onemklDiag diag, int64_t n,
                const double *a, double *x, int64_t incx);

void onemklCtpmv(sycl::queue device_queue, onemklUplo uplo,
                onemklTranspose trans, onemklDiag diag, int64_t n,
                const float _Complex *a, float _Complex *x, int64_t incx);

void onemklZtpmv(sycl::queue device_queue, onemklUplo uplo,
                onemklTranspose trans, onemklDiag diag, int64_t n,
                const double _Complex *a, double _Complex *x, int64_t incx);

void onemklStpsv(sycl::queue device_queue, onemklUplo uplo,
                onemklTranspose trans, onemklDiag diag, int64_t m,
                const float *a, float *x, int64_t incx);

void onemklDtpsv(sycl::queue device_queue, onemklUplo uplo,
                onemklTranspose trans, onemklDiag diag, int64_t m,
                const double *a, double *x, int64_t incx);

void onemklCtpsv(sycl::queue device_queue, onemklUplo uplo,
                onemklTranspose trans, onemklDiag diag, int64_t m,
                const float _Complex *a, float _Complex *x, int64_t incx);

void onemklZtpsv(sycl::queue device_queue, onemklUplo uplo,
                onemklTranspose trans, onemklDiag diag, int64_t m,
                const double _Complex *a, double _Complex *x, int64_t incx);

void onemklStrmv(sycl::queue device_queue, onemklUplo uplo, onemklTranspose trans,
                onemklDiag diag, int64_t n, const float *a, int64_t lda, float *x,
                int64_t incx);

void onemklDtrmv(sycl::queue device_queue, onemklUplo uplo, onemklTranspose trans,
                onemklDiag diag, int64_t n, const double *a, int64_t lda, double *x,
                int64_t incx);

void onemklCtrmv(sycl::queue device_queue, onemklUplo uplo, onemklTranspose trans,
                onemklDiag diag, int64_t n, const float _Complex *a, int64_t lda, float _Complex *x,
                int64_t incx);

void onemklZtrmv(sycl::queue device_queue, onemklUplo uplo, onemklTranspose trans,
                onemklDiag diag, int64_t n, const double _Complex *a, int64_t lda, double _Complex *x,
                int64_t incx);

// trsv
void onemklStrsv(sycl::queue device_queue, onemklUplo uplo, onemklTranspose trans,
                onemklDiag diag, int64_t n, const float *a, int64_t lda, float *x,
                int64_t incx);

void onemklDtrsv(sycl::queue device_queue, onemklUplo uplo, onemklTranspose trans,
                onemklDiag diag, int64_t n, const double *a, int64_t lda, double *x,
                int64_t incx);

void onemklCtrsv(sycl::queue device_queue, onemklUplo uplo, onemklTranspose trans,
                onemklDiag diag, int64_t n, const float _Complex *a, int64_t lda, float _Complex *x,
                int64_t incx);

void onemklZtrsv(sycl::queue device_queue, onemklUplo uplo, onemklTranspose trans,
                onemklDiag diag, int64_t n, const double _Complex *a, int64_t lda, double _Complex *x,
                int64_t incx);

void onemklCherk(sycl::queue device_queue, onemklUplo uplo, onemklTranspose trans, int64_t n, int64_t k,
                float alpha, const float _Complex* a, int64_t lda, float beta, float _Complex* c, int64_t ldc);
void onemklZherk(sycl::queue device_queue, onemklUplo uplo, onemklTranspose trans, int64_t n, int64_t k,
                double alpha, const double _Complex* a, int64_t lda, double beta, double _Complex* c, int64_t ldc);

void onemklCher2k(sycl::queue device_queue, onemklUplo uplo, onemklTranspose trans, int64_t n, int64_t k,
                float _Complex alpha, const float _Complex* a, int64_t lda, const float _Complex* b, int64_t ldb,
                float beta, float _Complex* c, int64_t ldc);
void onemklZher2k(sycl::queue device_queue, onemklUplo uplo, onemklTranspose trans, int64_t n, int64_t k,
                double _Complex alpha, const double _Complex* a, int64_t lda,  const double _Complex* b, int64_t ldb,
                double beta, double _Complex* c, int64_t ldc);

void onemklSsymm(sycl::queue device_queue, onemklSideMode side, onemklUplo uplo, int64_t m, int64_t n,
                float alpha, const float* a, int64_t lda, const float* b, int64_t ldb,
                float beta, float* c, int64_t ldc);
void onemklDsymm(sycl::queue device_queue, onemklSideMode side, onemklUplo uplo, int64_t m, int64_t n,
                double alpha, const double* a, int64_t lda, const double* b, int64_t ldb,
                double beta, double* c, int64_t ldc);
void onemklCsymm(sycl::queue device_queue, onemklSideMode side, onemklUplo uplo, int64_t m, int64_t n,
                float _Complex alpha, const float _Complex* a, int64_t lda, const float _Complex* b, int64_t ldb,
                float _Complex beta, float _Complex* c, int64_t ldc);
void onemklZsymm(sycl::queue device_queue, onemklSideMode side, onemklUplo uplo, int64_t m, int64_t n,
                double _Complex alpha, const double _Complex* a, int64_t lda, const double _Complex* b, int64_t ldb,
                double _Complex beta, double _Complex* c, int64_t ldc);

void onemklSsyrk(sycl::queue device_queue, onemklUplo uplo, onemklTranspose trans, int64_t n, int64_t k,
                float alpha, const float* a, int64_t lda, float beta, float* c, int64_t ldc);
void onemklDsyrk(sycl::queue device_queue, onemklUplo uplo, onemklTranspose trans, int64_t n, int64_t k,
                double alpha, const double* a, int64_t lda, double beta, double* c, int64_t ldc);
void onemklCsyrk(sycl::queue device_queue, onemklUplo uplo, onemklTranspose trans, int64_t n, int64_t k,
                float _Complex alpha, const float _Complex* a, int64_t lda, float _Complex beta, float _Complex* c, int64_t ldc);
void onemklZsyrk(sycl::queue device_queue, onemklUplo uplo, onemklTranspose trans, int64_t n, int64_t k,
                double _Complex alpha, const double _Complex* a, int64_t lda, double _Complex beta, double _Complex* c, int64_t ldc);

void onemklSsyr2k(sycl::queue device_queue, onemklUplo uplo, onemklTranspose trans, int64_t n, int64_t k,
                float alpha, const float* a, int64_t lda, const float* b, int64_t ldb, float beta, float* c, int64_t ldc);
void onemklDsyr2k(sycl::queue device_queue, onemklUplo uplo, onemklTranspose trans, int64_t n, int64_t k,
                double alpha, const double* a, int64_t lda, const double* b, int64_t ldb,double beta, double* c, int64_t ldc);
void onemklCsyr2k(sycl::queue device_queue, onemklUplo uplo, onemklTranspose trans, int64_t n, int64_t k,
                float _Complex alpha, const float _Complex* a, int64_t lda, const float _Complex* b, int64_t ldb,
                float _Complex beta, float _Complex* c, int64_t ldc);
void onemklZsyr2k(sycl::queue device_queue, onemklUplo uplo, onemklTranspose trans, int64_t n, int64_t k,
                double _Complex alpha, const double _Complex* a, int64_t lda, const double _Complex* b, int64_t ldb,
                double _Complex beta, double _Complex* c, int64_t ldc);

void onemklChemm(sycl::queue device_queue, onemklSideMode side, onemklUplo uplo, int64_t m, int64_t n,
                float _Complex alpha, const float _Complex* a, int64_t lda, const float _Complex* b, int64_t ldb,
                float _Complex beta, float _Complex* c, int64_t ldc);
void onemklZhemm(sycl::queue device_queue, onemklSideMode side, onemklUplo uplo, int64_t m, int64_t n,
                double _Complex alpha, const double _Complex* a, int64_t lda, const double _Complex* b, int64_t ldb,
                double _Complex beta, double _Complex* c, int64_t ldc);

void onemklStrmm(sycl::queue device_queue, onemklSideMode side, onemklUplo uplo, onemklTranspose trans, onemklDiag diag, int64_t m,
                int64_t n, float alpha, const float *a, int64_t lda, float *b, int64_t incb);
void onemklDtrmm(sycl::queue device_queue, onemklSideMode side, onemklUplo uplo, onemklTranspose trans, onemklDiag diag, int64_t m,
                int64_t n, double alpha, const double *a, int64_t lda, double *b, int64_t incb);
void onemklCtrmm(sycl::queue device_queue, onemklSideMode side, onemklUplo uplo, onemklTranspose trans, onemklDiag diag, int64_t m,
                int64_t n, float _Complex alpha, const float _Complex*a, int64_t lda, float _Complex*b, int64_t incb);
void onemklZtrmm(sycl::queue device_queue, onemklSideMode side, onemklUplo uplo, onemklTranspose trans, onemklDiag diag, int64_t m,
                int64_t n, double _Complex alpha, const double _Complex *a, int64_t lda, double _Complex *b, int64_t incb);

void onemklStrsm(sycl::queue device_queue, onemklSideMode side, onemklUplo uplo, onemklTranspose trans, onemklDiag diag, int64_t m,
                int64_t n, float alpha, const float *a, int64_t lda, float *b, int64_t incb);
void onemklDtrsm(sycl::queue device_queue, onemklSideMode side, onemklUplo uplo, onemklTranspose trans, onemklDiag diag, int64_t m,
                int64_t n, double alpha, const double *a, int64_t lda, double *b, int64_t incb);
void onemklCtrsm(sycl::queue device_queue, onemklSideMode side, onemklUplo uplo, onemklTranspose trans, onemklDiag diag, int64_t m,
                int64_t n, float _Complex alpha, const float _Complex*a, int64_t lda, float _Complex*b, int64_t incb);
void onemklZtrsm(sycl::queue device_queue, onemklSideMode side, onemklUplo uplo, onemklTranspose trans, onemklDiag diag, int64_t m,
                int64_t n, double _Complex alpha, const double _Complex *a, int64_t lda, double _Complex *b, int64_t incb);

// XXX: how to expose half in C?
// int onemklHgemm(sycl::queue device_queue, onemklTranspose transA,
//                onemklTranspose transB, int64_t m, int64_t n, int64_t k,
//                half alpha, const half *A, int64_t lda, const half *B,
//                int64_t ldb, half beta, half *C, int64_t ldc);
int onemklSgemm(sycl::queue device_queue, onemklTranspose transA,
                onemklTranspose transB, int64_t m, int64_t n, int64_t k,
                float alpha, const float *A, int64_t lda, const float *B,
                int64_t ldb, float beta, float *C, int64_t ldc);
int onemklDgemm(sycl::queue device_queue, onemklTranspose transA,
                onemklTranspose transB, int64_t m, int64_t n, int64_t k,
                double alpha, const double *A, int64_t lda, const double *B,
                int64_t ldb, double beta, double *C, int64_t ldc);
int onemklCgemm(sycl::queue device_queue, onemklTranspose transA,
                onemklTranspose transB, int64_t m, int64_t n, int64_t k,
                float _Complex alpha, const float _Complex *A, int64_t lda,
                const float _Complex *B, int64_t ldb, float _Complex beta,
                float _Complex *C, int64_t ldc);
int onemklZgemm(sycl::queue device_queue, onemklTranspose transA,
                onemklTranspose transB, int64_t m, int64_t n, int64_t k,
                double _Complex alpha, const double _Complex *A, int64_t lda,
                const double _Complex *B, int64_t ldb, double _Complex beta,
                double _Complex *C, int64_t ldc);


int64_t onemklDsyevd_scratchpad_size(sycl::queue device_queue,
				    onemklJob jobz,
				    onemklUplo upper_lower,
				    int64_t n, int64_t lda );

int64_t onemklDsyevd(sycl::queue device_queue,
		     onemklJob jobz, onemklUplo upper_lower,
		     int64_t n, double *a, int64_t lda,
		     double *w, double *scratchpad,
		     int64_t scratchpad_size);

  
void onemklDestroy();
#ifdef __cplusplus
}
#endif
