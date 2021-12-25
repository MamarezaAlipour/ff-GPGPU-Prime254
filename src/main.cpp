#include <bench_ntt.hpp>
#include <iomanip>
#include <iostream>
#include <test.hpp>

int main(int argc, char **argv) {
  sycl::device d{sycl::default_selector{}};
  sycl::queue q{d};
  std::cout << "running on " << d.get_info<sycl::info::device::name>() << "\n"
            << std::endl;

#if defined TEST // only run test cases !

  test_two_adic_root_of_unity();
  test_get_root_of_unity();
  std::cout << "passed prime field tests !" << std::endl;

  // check with rectangular matrix
  test_matrix_transposed_initialise(q, 1ul << 15, 1ul << 6);
  std::cout << "passed matrix transposed initialisation tests ! [rectangular]" << std::endl;
  // check with square matrix
  test_matrix_transposed_initialise(q, 1ul << 16, 1ul << 6);
  std::cout << "passed matrix transposed initialisation tests ! [square]" << std::endl;

  // takes square matrix of dim x dim size, transposes twice
  // finally asserts with original matrix
  test_matrix_transpose(q, 1ul << 10, 1ul << 6);
  std::cout << "passed matrix transposition tests !" << std::endl;

  test_twiddle_multiplication(q, 1ul << 15, 1ul << 6);
  std::cout << "passed twiddle multiplication tests ! [rectangular]" << std::endl;
  test_twiddle_multiplication(q, 1ul << 16, 1ul << 6);
  std::cout << "passed twiddle multiplication tests ! [square]" << std::endl;

  test_six_step_fft_ifft(q, 1ul << 17, 1ul << 6);
  std::cout << "passed fft/ifft tests !" << std::endl;

#elif defined BENCHMARK // only run benchmarks !

  // FFT benchmark variations ( based on whether data transfer cost is included
  // or not )

  std::cout << "\nSix-Step FFT (without data transfer cost)\n" << std::endl;
  std::cout << std::setw(11) << "dimension"
            << "\t\t" << std::setw(15) << "total" << std::endl;

  for (uint dim = 16; dim <= 23; dim++) {
    int64_t tm =
        benchmark_six_step_fft(q, 1ul << dim, 1 << 6, data_transfer_t::none);

    std::cout << std::setw(9) << std::right << (1ul << dim) << "\t\t"
              << std::setw(15) << std::right << (float)tm / 1000.f << " ms"
              << std::endl;
  }

  std::cout << "\nSix-Step FFT (with host -> device data transfer cost)\n"
            << std::endl;
  std::cout << std::setw(11) << "dimension"
            << "\t\t" << std::setw(15) << "total" << std::endl;

  for (uint dim = 16; dim <= 23; dim++) {
    int64_t tm = benchmark_six_step_fft(q, 1ul << dim, 1 << 6,
                                        data_transfer_t::only_host_to_device);

    std::cout << std::setw(9) << std::right << (1ul << dim) << "\t\t"
              << std::setw(15) << std::right << (float)tm / 1000.f << " ms"
              << std::endl;
  }

  std::cout << "\nSix-Step FFT (with device -> host data transfer cost)\n"
            << std::endl;
  std::cout << std::setw(11) << "dimension"
            << "\t\t" << std::setw(15) << "total" << std::endl;

  for (uint dim = 16; dim <= 23; dim++) {
    int64_t tm = benchmark_six_step_fft(q, 1ul << dim, 1 << 6,
                                        data_transfer_t::only_device_to_host);

    std::cout << std::setw(9) << std::right << (1ul << dim) << "\t\t"
              << std::setw(15) << std::right << (float)tm / 1000.f << " ms"
              << std::endl;
  }

  std::cout << "\nSix-Step FFT (with host <-> device data transfer cost)\n"
            << std::endl;
  std::cout << std::setw(11) << "dimension"
            << "\t\t" << std::setw(15) << "total" << std::endl;

  for (uint dim = 16; dim <= 23; dim++) {
    int64_t tm =
        benchmark_six_step_fft(q, 1ul << dim, 1 << 6, data_transfer_t::both);

    std::cout << std::setw(9) << std::right << (1ul << dim) << "\t\t"
              << std::setw(15) << std::right << (float)tm / 1000.f << " ms"
              << std::endl;
  }

  // IFFT benchmark variations ( based on whether data transfer cost is included
  // or not )

  std::cout << "\nSix-Step IFFT (without data transfer cost)\n" << std::endl;
  std::cout << std::setw(11) << "dimension"
            << "\t\t" << std::setw(15) << "total" << std::endl;

  for (uint dim = 16; dim <= 23; dim++) {
    int64_t tm =
        benchmark_six_step_ifft(q, 1ul << dim, 1 << 6, data_transfer_t::none);

    std::cout << std::setw(9) << std::right << (1ul << dim) << "\t\t"
              << std::setw(15) << std::right << (float)tm / 1000.f << " ms"
              << std::endl;
  }

  std::cout << "\nSix-Step IFFT (with host -> device data transfer cost)\n"
            << std::endl;
  std::cout << std::setw(11) << "dimension"
            << "\t\t" << std::setw(15) << "total" << std::endl;

  for (uint dim = 16; dim <= 23; dim++) {
    int64_t tm = benchmark_six_step_ifft(q, 1ul << dim, 1 << 6,
                                         data_transfer_t::only_host_to_device);

    std::cout << std::setw(9) << std::right << (1ul << dim) << "\t\t"
              << std::setw(15) << std::right << (float)tm / 1000.f << " ms"
              << std::endl;
  }

  std::cout << "\nSix-Step IFFT (with device -> host data transfer cost)\n"
            << std::endl;
  std::cout << std::setw(11) << "dimension"
            << "\t\t" << std::setw(15) << "total" << std::endl;

  for (uint dim = 16; dim <= 23; dim++) {
    int64_t tm = benchmark_six_step_ifft(q, 1ul << dim, 1 << 6,
                                         data_transfer_t::only_device_to_host);

    std::cout << std::setw(9) << std::right << (1ul << dim) << "\t\t"
              << std::setw(15) << std::right << (float)tm / 1000.f << " ms"
              << std::endl;
  }

  std::cout << "\nSix-Step IFFT (with host <-> device data transfer cost)\n"
            << std::endl;
  std::cout << std::setw(11) << "dimension"
            << "\t\t" << std::setw(15) << "total" << std::endl;

  for (uint dim = 16; dim <= 23; dim++) {
    int64_t tm =
        benchmark_six_step_ifft(q, 1ul << dim, 1 << 6, data_transfer_t::both);

    std::cout << std::setw(9) << std::right << (1ul << dim) << "\t\t"
              << std::setw(15) << std::right << (float)tm / 1000.f << " ms"
              << std::endl;
  }

#else // do nothing useful !
  std::cout << "Check https://github.com/PariKhaleghi/ff-GPGPU-Prime254/blob/master/src/Makefile#L8-L19" << std::endl;
#endif

  return 0;
}
