#include <test.hpp>

void test_two_adic_root_of_unity() {
  ff_p254_t one(1_ZL);
  ff_p254_t mod(mod_p254);
  ff_p254_t t(268435456_ZL); // 1 << 28
  ff_p254_t k = (mod - one) / t;

  assert(static_cast<ff_p254_t>(cbn::mod_exp(
             GENERATOR.data, k.data, mod_p254_bn)) == TWO_ADIC_ROOT_OF_UNITY);
}

void test_get_root_of_unity() {
  ff_p254_t one(1_ZL);
  ff_p254_t t_1(268435456_ZL); // 1 << 28
  ff_p254_t t_2(134217728_ZL); // 1 << 27
  ff_p254_t root_28 = get_root_of_unity(28);

  assert(root_28 == TWO_ADIC_ROOT_OF_UNITY);
  assert(static_cast<ff_p254_t>(
             cbn::mod_exp(root_28.data, t_1.data, mod_p254_bn)) == one);

  ff_p254_t root_27 = get_root_of_unity(27);
  ff_p254_t exp = static_cast<ff_p254_t>(
      cbn::mod_exp(root_28.data, cbn::to_big_int(2_ZL), mod_p254_bn));

  assert(root_27 == exp);
  assert(static_cast<ff_p254_t>(
             cbn::mod_exp(root_27.data, t_2.data, mod_p254_bn)) == one);
}

void test_matrix_transposed_initialise(sycl::queue &q, const uint64_t dim,
                                       const uint64_t wg_size) {
  assert((dim & (dim - 1ul)) == 0);

  uint64_t log_2_dim = (uint64_t)sycl::log2((float)dim);
  uint64_t n1 = 1 << (log_2_dim / 2);
  uint64_t n2 = dim / n1;
  uint64_t n = std::max(n1, n2);

  assert(n1 == n2 || n2 == 2 * n1);
  assert(log_2_dim > 0 && log_2_dim <= TWO_ADICITY_);

  // n1 x n2 matrix
  ff_p254_t *vec_src =
      static_cast<ff_p254_t *>(sycl::malloc_shared(sizeof(ff_p254_t) * dim, q));
  // n x n matrix, padded with zeros if n2 > n1
  ff_p254_t *vec_dst = static_cast<ff_p254_t *>(
      sycl::malloc_shared(sizeof(ff_p254_t) * n * n, q));

  sycl::event evt_0 = q.memset(vec_dst, 0, sizeof(ff_p254_t) * n * n);
  prepare_random_vector(vec_src, dim);
  matrix_transposed_initialise(q, vec_src, vec_dst, n2, n1, n, wg_size, {evt_0})
      .wait();

  for (uint64_t i = 0; i < n2; i++) {
    for (uint64_t j = 0; j < n1; j++) {
      // if source matrix is of  2 x 4
      // then destination should become 4 x 2
      // but notice, we I'm using one width variable, with is
      // set of max(n1, n2) = 4, in this context
      // so destination matrix is of actually dimension 4 x 4
      // so that it won't have to do any more memory allocation
      // when transposing one rectangular matrix
      assert(*(vec_src + j * n2 + i) == *(vec_dst + i * n + j));
    }
  }

  sycl::free(vec_src, q);
  sycl::free(vec_dst, q);
}

void test_matrix_transpose(sycl::queue &q, const uint64_t dim,
                           const uint64_t wg_size) {
  ff_p254_t *vec_a = static_cast<ff_p254_t *>(
      sycl::malloc_shared(sizeof(ff_p254_t) * dim * dim, q));
  ff_p254_t *vec_b = static_cast<ff_p254_t *>(
      sycl::malloc_shared(sizeof(ff_p254_t) * dim * dim, q));

  prepare_random_vector(vec_a, dim * dim);
  sycl::event evt_0 = q.memcpy(vec_b, vec_a, sizeof(ff_p254_t) * dim * dim);

  sycl::event evt_1 = matrix_transpose(q, vec_b, dim, {evt_0});
  sycl::event evt_2 = matrix_transpose(q, vec_b, dim, {evt_1});
  evt_2.wait();

  for (uint64_t i = 0; i < dim * dim; i++) {
    assert(*(vec_a + i) == *(vec_b + i));
  }

  sycl::free(vec_a, q);
  sycl::free(vec_b, q);
}

void test_twiddle_multiplication(sycl::queue &q, const uint64_t dim,
                                 const uint64_t wg_size) {
  assert((dim & (dim - 1ul)) == 0);

  uint64_t log_2_dim = (uint64_t)sycl::log2((float)dim);
  uint64_t n1 = 1 << (log_2_dim / 2);
  uint64_t n2 = dim / n1;
  uint64_t n = sycl::max(n1, n2);

  assert(n1 == n2 || n2 == 2 * n1);
  assert(log_2_dim > 0 && log_2_dim <= TWO_ADICITY_);

  ff_p254_t *vec_src = static_cast<ff_p254_t *>(
      sycl::malloc_shared(sizeof(ff_p254_t) * n * n, q));
  ff_p254_t *vec_dst =
      static_cast<ff_p254_t *>(sycl::malloc_host(sizeof(ff_p254_t) * n * n, q));
  ff_p254_t *twiddles =
      static_cast<ff_p254_t *>(sycl::malloc_device(sizeof(ff_p254_t) * n, q));
  ff_p254_t *omega =
      static_cast<ff_p254_t *>(sycl::malloc_shared(sizeof(ff_p254_t), q));

  prepare_random_vector(vec_src, n * n);

  sycl::event evt_0 =
      q.single_task([=]() { *omega = get_root_of_unity(log_2_dim); });
  sycl::event evt_1 = q.memcpy(vec_dst, vec_src, sizeof(ff_p254_t) * n * n);
  sycl::event evt_2 = twiddle_multiplication(q, vec_src, omega, n2, n1, n,
                                             wg_size, {evt_0, evt_1});

  evt_2.wait();

  for (uint64_t i = 0; i < n2; i++) {
    for (uint64_t j = 0; j < n1; j++) {
      assert(*(vec_src + i * n + j) ==
             *(vec_dst + i * n + j) *
                 static_cast<ff_p254_t>(cbn::mod_exp(
                     (*omega).data, ff_p254_t(i * j).data, mod_p254_bn)));
    }
  }

  sycl::free(vec_src, q);
  sycl::free(vec_dst, q);
  sycl::free(twiddles, q);
  sycl::free(omega, q);
}

void test_six_step_fft_ifft(sycl::queue &q, const uint64_t dim,
                            const uint64_t wg_size) {
  assert((dim & (dim - 1ul)) == 0);

  uint64_t log_2_dim = (uint64_t)sycl::log2((float)dim);
  uint64_t n1 = 1 << (log_2_dim / 2);
  uint64_t n2 = dim / n1;
  uint64_t n = sycl::max(n1, n2);

  assert(log_2_dim > 0 && log_2_dim <= TWO_ADICITY_);

  ff_p254_t *vec_src =
      static_cast<ff_p254_t *>(sycl::malloc_host(sizeof(ff_p254_t) * dim, q));
  ff_p254_t *vec_fwd =
      static_cast<ff_p254_t *>(sycl::malloc_device(sizeof(ff_p254_t) * dim, q));
  ff_p254_t *vec_scratch = static_cast<ff_p254_t *>(
      sycl::malloc_device(sizeof(ff_p254_t) * n * n, q));
  ff_p254_t *vec_inv =
      static_cast<ff_p254_t *>(sycl::malloc_shared(sizeof(ff_p254_t) * dim, q));
  ff_p254_t *omega =
      static_cast<ff_p254_t *>(sycl::malloc_device(sizeof(ff_p254_t) * 4, q));

  prepare_random_vector(vec_src, dim);

  sycl::event evt_0 = q.memcpy(vec_fwd, vec_src, sizeof(ff_p254_t) * dim);
  sycl::event evt_1 = six_step_fft(q, vec_fwd, vec_scratch, omega + 0,
                                   omega + 1, omega + 2, dim, wg_size, {evt_0});

  sycl::event evt_2 = q.submit([&](sycl::handler &h) {
    h.depends_on(evt_1);
    h.memcpy(vec_inv, vec_fwd, sizeof(ff_p254_t) * dim);
  });
  sycl::event evt_3 =
      six_step_ifft(q, vec_inv, vec_scratch, omega + 0, omega + 1, omega + 2,
                    omega + 3, dim, wg_size, {evt_2});

  evt_3.wait();

  for (uint64_t i = 0; i < dim; i++) {
    assert(*(vec_src + i) == *(vec_inv + i));
  }

  sycl::free(vec_src, q);
  sycl::free(vec_fwd, q);
  sycl::free(vec_scratch, q);
  sycl::free(vec_inv, q);
  sycl::free(omega, q);
}
