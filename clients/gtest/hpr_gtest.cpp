/* ************************************************************************
 * Copyright (C) 2016-2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * ************************************************************************ */

#include "testing_hpr.hpp"
#include "testing_hpr_batched.hpp"
#include "testing_hpr_strided_batched.hpp"
#include "utility.h"
#include <math.h>
#include <stdexcept>
#include <vector>

using std::vector;
using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;

// only GCC/VS 2010 comes with std::tr1::tuple, but it is unnecessary,  std::tuple is good enough;

typedef std::tuple<int, int, double, char, double, int, bool> hpr_tuple;

/* =====================================================================
README: This file contains testers to verify the correctness of
        BLAS routines with google test

        It is supposed to be played/used by advance / expert users
        Normal users only need to get the library routines without testers
     =================================================================== */

/* =====================================================================
Advance users only: BrainStorm the parameters but do not make artificial one which invalidates the
matrix.
like lda pairs with M, and "lda must >= M". case "lda < M" will be guarded by argument-checkers
inside API of course.
Yet, the goal of this file is to verify result correctness not argument-checkers.

Representative sampling is sufficient, endless brute-force sampling is not necessary
=================================================================== */

// vector of vector, each vector is a {N};
// add/delete as a group
const vector<int> matrix_size_range = {-1, 11, 16, 32, 65};

// vector of vector, each pair is a {incx};
// add/delete this list in pairs, like {1}
const vector<int> incx_incy_range = {-1, 0, 2};

// vector, each entry is  {alpha};
// add/delete single values, like {2.0}
const vector<double> alpha_range = {-0.5, 2.0, 0.0};

const vector<char> uplo_range = {
    'L',
    'U',
};

const vector<double> stride_scale_range = {1.0, 2.5};
const vector<int>    batch_count_range  = {-1, 0, 1, 2, 10};

const bool is_fortran[] = {false, true};

/* ===============Google Unit Test==================================================== */

/* =====================================================================
     BLAS-2 hpr:
=================================================================== */

/* ============================Setup Arguments======================================= */

// Please use "class Arguments" (see utility.hpp) to pass parameters to templated testers;
// Some routines may not touch/use certain "members" of objects "arg".
// like BLAS-1 Scal does not have lda, BLAS-2 GEMV does not have ldb, ldc;
// That is fine. These testers & routines will leave untouched members alone.
// Do not use std::tuple to directly pass parameters to testers
// by std:tuple, you have unpack it with extreme care for each one by like "std::get<0>" which is
// not intuitive and error-prone

Arguments setup_hpr_arguments(hpr_tuple tup)
{
    Arguments arg;

    arg.N            = std::get<0>(tup);
    arg.incx         = std::get<1>(tup);
    arg.alpha        = std::get<2>(tup);
    arg.uplo         = std::get<3>(tup);
    arg.stride_scale = std::get<4>(tup);
    arg.batch_count  = std::get<5>(tup);
    arg.fortran      = std::get<6>(tup);
    arg.timing       = 0;

    return arg;
}

class blas2_hpr_gtest : public ::TestWithParam<hpr_tuple>
{
protected:
    blas2_hpr_gtest() {}
    virtual ~blas2_hpr_gtest() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

// hpr
TEST_P(blas2_hpr_gtest, hpr_gtest_float)
{
    // GetParam return a tuple. Tee setup routine unpack the tuple
    // and initializes arg(Arguments) which will be passed to testing routine
    // The Arguments data struture have physical meaning associated.
    // while the tuple is non-intuitive.

    Arguments arg = setup_hpr_arguments(GetParam());

    hipblasStatus_t status = testing_hpr<hipblasComplex>(arg);

    // if not success, then the input argument is problematic, so detect the error message
    if(status != HIPBLAS_STATUS_SUCCESS)
    {
        if(arg.N < 0 || arg.incx == 0)
        {
            EXPECT_EQ(HIPBLAS_STATUS_INVALID_VALUE, status);
        }
        else
        {
            EXPECT_EQ(HIPBLAS_STATUS_SUCCESS, status); // fail
        }
    }
}

TEST_P(blas2_hpr_gtest, hpr_gtest_double)
{
    // GetParam return a tuple. Tee setup routine unpack the tuple
    // and initializes arg(Arguments) which will be passed to testing routine
    // The Arguments data struture have physical meaning associated.
    // while the tuple is non-intuitive.

    Arguments arg = setup_hpr_arguments(GetParam());

    hipblasStatus_t status = testing_hpr<hipblasDoubleComplex>(arg);

    // if not success, then the input argument is problematic, so detect the error message
    if(status != HIPBLAS_STATUS_SUCCESS)
    {
        if(arg.N < 0 || arg.incx == 0)
        {
            EXPECT_EQ(HIPBLAS_STATUS_INVALID_VALUE, status);
        }
        else
        {
            EXPECT_EQ(HIPBLAS_STATUS_SUCCESS, status); // fail
        }
    }
}

#if !defined(__HIP_PLATFORM_NVCC__) && !defined(__HIP_PLATFORM_ONEAPI__)

// hpr_batched
TEST_P(blas2_hpr_gtest, hpr_batched_gtest_float)
{
    // GetParam return a tuple. Tee setup routine unpack the tuple
    // and initializes arg(Arguments) which will be passed to testing routine
    // The Arguments data struture have physical meaning associated.
    // while the tuple is non-intuitive.

    Arguments arg = setup_hpr_arguments(GetParam());

    hipblasStatus_t status = testing_hpr_batched<hipblasComplex>(arg);

    // if not success, then the input argument is problematic, so detect the error message
    if(status != HIPBLAS_STATUS_SUCCESS)
    {
        if(arg.N < 0 || arg.incx == 0 || arg.batch_count < 0)
        {
            EXPECT_EQ(HIPBLAS_STATUS_INVALID_VALUE, status);
        }
        else
        {
            EXPECT_EQ(HIPBLAS_STATUS_SUCCESS, status); // fail
        }
    }
}

TEST_P(blas2_hpr_gtest, hpr_batched_gtest_double)
{
    // GetParam return a tuple. Tee setup routine unpack the tuple
    // and initializes arg(Arguments) which will be passed to testing routine
    // The Arguments data struture have physical meaning associated.
    // while the tuple is non-intuitive.

    Arguments arg = setup_hpr_arguments(GetParam());

    hipblasStatus_t status = testing_hpr_batched<hipblasDoubleComplex>(arg);

    // if not success, then the input argument is problematic, so detect the error message
    if(status != HIPBLAS_STATUS_SUCCESS)
    {
        if(arg.N < 0 || arg.incx == 0 || arg.batch_count < 0)
        {
            EXPECT_EQ(HIPBLAS_STATUS_INVALID_VALUE, status);
        }
        else
        {
            EXPECT_EQ(HIPBLAS_STATUS_SUCCESS, status); // fail
        }
    }
}

// hpr_strided_batched
TEST_P(blas2_hpr_gtest, hpr_strided_batched_gtest_float)
{
    // GetParam return a tuple. Tee setup routine unpack the tuple
    // and initializes arg(Arguments) which will be passed to testing routine
    // The Arguments data struture have physical meaning associated.
    // while the tuple is non-intuitive.

    Arguments arg = setup_hpr_arguments(GetParam());

    hipblasStatus_t status = testing_hpr_strided_batched<hipblasComplex>(arg);

    // if not success, then the input argument is problematic, so detect the error message
    if(status != HIPBLAS_STATUS_SUCCESS)
    {
        if(arg.N < 0 || arg.incx == 0 || arg.batch_count < 0)
        {
            EXPECT_EQ(HIPBLAS_STATUS_INVALID_VALUE, status);
        }
        else
        {
            EXPECT_EQ(HIPBLAS_STATUS_SUCCESS, status); // fail
        }
    }
}

TEST_P(blas2_hpr_gtest, hpr_strided_batched_gtest_double)
{
    // GetParam return a tuple. Tee setup routine unpack the tuple
    // and initializes arg(Arguments) which will be passed to testing routine
    // The Arguments data struture have physical meaning associated.
    // while the tuple is non-intuitive.

    Arguments arg = setup_hpr_arguments(GetParam());

    hipblasStatus_t status = testing_hpr_strided_batched<hipblasDoubleComplex>(arg);

    // if not success, then the input argument is problematic, so detect the error message
    if(status != HIPBLAS_STATUS_SUCCESS)
    {
        if(arg.N < 0 || arg.incx == 0 || arg.batch_count < 0)
        {
            EXPECT_EQ(HIPBLAS_STATUS_INVALID_VALUE, status);
        }
        else
        {
            EXPECT_EQ(HIPBLAS_STATUS_SUCCESS, status); // fail
        }
    }
}

#endif

// notice we are using vector of vector
// so each elment in xxx_range is a avector,
// ValuesIn take each element (a vector) and combine them and feed them to test_p
// The combinations are  { {M, N, lda}, {incx,incy} {alpha} }

INSTANTIATE_TEST_SUITE_P(hipblasHpr,
                         blas2_hpr_gtest,
                         Combine(ValuesIn(matrix_size_range),
                                 ValuesIn(incx_incy_range),
                                 ValuesIn(alpha_range),
                                 ValuesIn(uplo_range),
                                 ValuesIn(stride_scale_range),
                                 ValuesIn(batch_count_range),
                                 ValuesIn(is_fortran)));
