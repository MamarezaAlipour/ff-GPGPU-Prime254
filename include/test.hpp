#pragma once
#include <cassert>
#include <ntt.hpp>
#include <utils.hpp>

void test_two_adic_root_of_unity();
void test_get_root_of_unity();

void test_matrix_transposed_initialise(sycl::queue &q, const uint64_t dim,
                                       const uint64_t wg_size);

void test_matrix_transpose(sycl::queue &q, const uint64_t dim,
                           const uint64_t wg_size);

void test_twiddle_multiplication(sycl::queue &q, const uint64_t dim,
                                 const uint64_t wg_size);

void test_six_step_fft_ifft(sycl::queue &q, const uint64_t dim,
                            const uint64_t wg_size);
