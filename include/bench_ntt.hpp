#pragma once
#include <ntt.hpp>
#include <utils.hpp>

enum data_transfer_t { only_host_to_device, only_device_to_host, both, none };

int64_t benchmark_six_step_fft(sycl::queue &q, const uint64_t dim,
                               const uint64_t wg_size, data_transfer_t choice);

int64_t benchmark_six_step_ifft(sycl::queue &q, const uint64_t dim,
                                const uint64_t wg_size, data_transfer_t choice);
