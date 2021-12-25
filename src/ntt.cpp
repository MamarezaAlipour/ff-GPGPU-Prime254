#include <ntt.hpp>

ff_p254_t get_root_of_unity(uint64_t n) {
  uint64_t pow_ = 1ul << (28 - n);
  ff_p254_t pow(pow_);

  return static_cast<ff_p254_t>(
      cbn::mod_exp(TWO_ADIC_ROOT_OF_UNITY.data, pow.data, mod_p254_bn));
}

sycl::event matrix_transposed_initialise(
    sycl::queue &q, ff_p254_t *vec_src, ff_p254_t *vec_dst, const uint64_t rows,
    const uint64_t cols, const uint64_t width, const uint64_t wg_size,
    std::vector<sycl::event> evts) {
  return q.submit([&](sycl::handler &h) {
    h.depends_on(evts);

    h.parallel_for<class kernelMatrixTransposedInitialise>(
        sycl::nd_range<2>{sycl::range<2>{rows, cols},
                          sycl::range<2>{1, wg_size}},
        [=](sycl::nd_item<2> it) [[intel::reqd_sub_group_size(32)]] {
          sycl::sub_group sg = it.get_sub_group();

          const size_t r = it.get_global_id(0);
          const size_t c = it.get_global_id(1);

          const uint64_t width_ = sycl::group_broadcast(sg, width);

          *(vec_dst + r * width_ + c) = *(vec_src + c * width_ + r);
        });
  });
}

sycl::event matrix_transpose(sycl::queue &q, ff_p254_t *data,
                             const uint64_t dim,
                             std::vector<sycl::event> evts) {
  constexpr size_t TILE_DIM = 1 << 4;
  constexpr size_t BLOCK_ROWS = 1 << 3;

  assert(TILE_DIM >= BLOCK_ROWS);

  return q.submit([&](sycl::handler &h) {
    sycl::accessor<ff_p254_t, 2, sycl::access_mode::read_write,
                   sycl::target::local>
        tile_s{sycl::range<2>{TILE_DIM, TILE_DIM + 1}, h};
    sycl::accessor<ff_p254_t, 2, sycl::access_mode::read_write,
                   sycl::target::local>
        tile_d{sycl::range<2>{TILE_DIM, TILE_DIM + 1}, h};

    h.depends_on(evts);
    h.parallel_for<class kernelMatrixTransposition>(
        sycl::nd_range<2>{sycl::range<2>{dim / (TILE_DIM / BLOCK_ROWS), dim},
                          sycl::range<2>{BLOCK_ROWS, TILE_DIM}},
        [=](sycl::nd_item<2> it) [[intel::reqd_sub_group_size(16)]] {
          sycl::group<2> grp = it.get_group();
          const size_t grp_id_x = it.get_group().get_id(1);
          const size_t grp_id_y = it.get_group().get_id(0);
          const size_t loc_id_x = it.get_local_id(1);
          const size_t loc_id_y = it.get_local_id(0);
          const size_t grp_width_x = it.get_group().get_group_range(1);

          // @note x denotes index along x-axis
          // while y denotes index along y-axis
          //
          // so in usual (row, col) indexing of 2D array
          // row = y, col = x
          const size_t x = grp_id_x * TILE_DIM + loc_id_x;
          const size_t y = grp_id_y * TILE_DIM + loc_id_y;

          const size_t width = grp_width_x * TILE_DIM;

          // non-diagonal cell blocks
          if (grp_id_y > grp_id_x) {
            size_t dx = grp_id_y * TILE_DIM + loc_id_x;
            size_t dy = grp_id_x * TILE_DIM + loc_id_y;

            for (size_t j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
              tile_s[loc_id_y + j][loc_id_x] = *(data + (y + j) * width + x);
            }

            for (size_t j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
              tile_d[loc_id_y + j][loc_id_x] = *(data + (dy + j) * width + dx);
            }

            sycl::group_barrier(grp, sycl::memory_scope::work_group);

            for (size_t j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
              *(data + (dy + j) * width + dx) = tile_s[loc_id_x][loc_id_y + j];
            }

            for (size_t j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
              *(data + (y + j) * width + x) = tile_d[loc_id_x][loc_id_y + j];
            }

            return;
          }

          // diagonal cell blocks
          if (grp_id_y == grp_id_x) {
            for (size_t j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
              tile_s[loc_id_y + j][loc_id_x] = *(data + (y + j) * width + x);
            }

            sycl::group_barrier(grp, sycl::memory_scope::work_group);

            for (size_t j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
              *(data + (y + j) * width + x) = tile_s[loc_id_x][loc_id_y + j];
            }
          }
        });
  });
}

sycl::event twiddle_multiplication(sycl::queue &q, ff_p254_t *vec,
                                   ff_p254_t *omega, const uint64_t rows,
                                   const uint64_t cols, const uint64_t width,
                                   const uint64_t wg_size,
                                   std::vector<sycl::event> evts) {
  assert(cols == width || 2 * cols == width);

  return q.submit([&](sycl::handler &h) {
    sycl::accessor<ff_p254_t, 1, sycl::access_mode::read_write,
                   sycl::target::local>
        lds{sycl::range<1>{1}, h};

    h.depends_on(evts);
    h.parallel_for<class kernelTwiddleMultiplication>(
        sycl::nd_range<2>{sycl::range<2>{rows, cols},
                          sycl::range<2>{1, wg_size}},
        [=](sycl::nd_item<2> it) [[intel::reqd_sub_group_size(16)]] {
          const uint64_t r = it.get_global_id(0);
          const uint64_t c = it.get_global_id(1);
          sycl::group<2> grp = it.get_group();

          // only work-group leader helps in caching
          // twiddle in local memory
          if (it.get_local_linear_id() == 0) {
            lds[0] = *omega;
          }

          // until all work-items of this work-group
          // arrives here, wait !
          sycl::group_barrier(grp);

          // after that all work-items of work-group reads from cached twiddle
          // from local memory
          *(vec + r * width + c) *= static_cast<ff_p254_t>(
              cbn::mod_exp(lds[0].data, ff_p254_t(r * c).data, mod_p254_bn));
        });
  });
}

sycl::event row_wise_transform(sycl::queue &q, ff_p254_t *vec, ff_p254_t *omega,
                               const uint64_t rows, const uint64_t cols,
                               const uint64_t width, const uint64_t wg_size,
                               std::vector<sycl::event> evts) {
  uint64_t log_2_dim = (uint64_t)sycl::log2((float)cols);

  std::vector<sycl::event> _evts;
  _evts.reserve(log_2_dim);

  // if you change this number, make sure
  // you also change `[[intel::reqd_sub_group_size(Z)]]`
  // below, such that SUBGROUP_SIZE == Z
  constexpr uint64_t SUBGROUP_SIZE = 1ul << 5;

  assert((SUBGROUP_SIZE & (SUBGROUP_SIZE - 1ul)) == 0ul &&
         (SUBGROUP_SIZE <= (1ul << 6)));
  assert((wg_size % SUBGROUP_SIZE) == 0);

  for (int64_t i = log_2_dim - 1ul; i >= 0; i--) {
    sycl::event evt = q.submit([&](sycl::handler &h) {
      if (i == log_2_dim - 1ul) {
        // only first submission depends on
        // previous kernel executions, whose events
        // are passed as argument to this function
        h.depends_on(evts);
      } else {
        // all next kernel submissions
        // depend on just previous kernel submission
        // from body of this loop
        h.depends_on(_evts.at(log_2_dim - (i + 2)));
      }

      h.parallel_for<class kernelCooleyTukeyRowWiseFFT>(
          sycl::nd_range<2>{sycl::range<2>{rows, cols},
                            sycl::range<2>{1, wg_size}},
          [=](sycl::nd_item<2> it) [[intel::reqd_sub_group_size(16)]] {
            const uint64_t r = it.get_global_id(0);
            const uint64_t k = it.get_global_id(1);
            const uint64_t p = 1ul << i;
            const uint64_t q = cols / p;

            uint64_t k_rev = bit_rev(k, log_2_dim) % q;
            ff_p254_t ω = static_cast<ff_p254_t>(cbn::mod_exp(
                (*omega).data, ff_p254_t(p * k_rev).data, mod_p254_bn));

            if (k < (k ^ p)) {
              ff_p254_t tmp_k = *(vec + r * width + k);
              ff_p254_t tmp_k_p = *(vec + r * width + (k ^ p));
              ff_p254_t tmp_k_p_ω = tmp_k_p * ω;

              *(vec + r * width + k) = tmp_k + tmp_k_p_ω;
              *(vec + r * width + (k ^ p)) = tmp_k - tmp_k_p_ω;
            }
          });
    });
    _evts.push_back(evt);
  }

  return q.submit([&](sycl::handler &h) {
    // final reordering kernel depends on very
    // last kernel submission performed in above loop
    h.depends_on(_evts.at(log_2_dim - 1));
    h.parallel_for<class kernelCooleyTukeyRowWiseFFTFinalReorder>(
        sycl::nd_range<2>{sycl::range<2>{rows, cols},
                          sycl::range<2>{1, wg_size}},
        [=](sycl::nd_item<2> it) [[intel::reqd_sub_group_size(16)]] {
          const uint64_t r = it.get_global_id(0);
          const uint64_t k = it.get_global_id(1);
          const uint64_t k_perm = permute_index(k, cols);

          if (k_perm > k) {
            ff_p254_t a = *(vec + r * width + k);
            ff_p254_t b = *(vec + r * width + k_perm);

            *(vec + r * width + k) = b;
            *(vec + r * width + k_perm) = a;
          }
        });
  });
}

uint64_t bit_rev(uint64_t v, uint64_t max_bit_width) {
  uint64_t v_rev = 0ul;
  for (uint64_t i = 0; i < max_bit_width; i++) {
    v_rev += ((v >> i) & 0b1) * (1ul << (max_bit_width - 1ul - i));
  }
  return v_rev;
}

uint64_t rev_all_bits(uint64_t n) {
  uint64_t rev = 0;

  for (uint8_t i = 0; i < 64; i++) {
    if ((1ul << i) & n) {
      rev |= (1ul << (63 - i));
    }
  }

  return rev;
}

uint64_t permute_index(uint64_t idx, uint64_t size) {
  if (size == 1ul) {
    return 0ul;
  }

  uint64_t bits = sycl::ext::intel::ctz(size);
  return rev_all_bits(idx) >> (64ul - bits);
}

sycl::event six_step_fft(sycl::queue &q, ff_p254_t *vec, ff_p254_t *vec_scratch,
                         ff_p254_t *omega_dim, ff_p254_t *omega_n1,
                         ff_p254_t *omega_n2, const uint64_t dim,
                         const uint64_t wg_size,
                         std::vector<sycl::event> evts) {
  assert((dim & (dim - 1ul)) == 0);

  uint64_t log_2_dim = (uint64_t)sycl::log2((float)dim);

  uint64_t n1 = 1 << (log_2_dim / 2);
  uint64_t n2 = dim / n1;
  uint64_t n = sycl::max(n1, n2);

  uint64_t log_2_n1 = (uint64_t)sycl::log2((float)n1);
  uint64_t log_2_n2 = (uint64_t)sycl::log2((float)n2);

  assert(n1 == n2 || n2 == 2 * n1);
  assert(log_2_dim > 0 && log_2_dim <= TWO_ADICITY_);

  // compute i-th root of unity, where i = {dim, n1, n2}
  sycl::event evt_0 = q.submit([&](sycl::handler &h) {
    h.depends_on(evts);
    h.single_task([=]() {
      *omega_dim = get_root_of_unity(log_2_dim);
      *omega_n1 = get_root_of_unity(log_2_n1);
      *omega_n2 = get_root_of_unity(log_2_n2);
    });
  });

  // Step 1: Transpose Matrix
  sycl::event evt_1 = matrix_transposed_initialise(q, vec, vec_scratch, n2, n1,
                                                   n, wg_size, evts);

  // Step 2: n2-many parallel n1-point Cooley-Tukey style NTT
  sycl::event evt_2 = row_wise_transform(q, vec_scratch, omega_n1, n2, n1, n,
                                         wg_size, {evt_0, evt_1});

  // Step 3: Multiply by twiddle factors
  sycl::event evt_3 = twiddle_multiplication(q, vec_scratch, omega_dim, n2, n1,
                                             n, wg_size, {evt_2});

  // Step 4: Transpose Matrix
  sycl::event evt_4 = matrix_transpose(q, vec_scratch, n, {evt_3});

  // Step 5: n1-many parallel n2-point Cooley-Tukey NTT
  sycl::event evt_5 =
      row_wise_transform(q, vec_scratch, omega_n2, n1, n2, n, wg_size, {evt_4});

  // Step 6: Transpose Matrix
  sycl::event evt_6 = matrix_transpose(q, vec_scratch, n, {evt_5});

  // copy result back to source vector
  return q.submit([&](sycl::handler &h) {
    h.depends_on(evt_6);
    h.parallel_for<class kernelFFTCopyBack>(
        sycl::nd_range<2>{sycl::range<2>{n2, n1}, sycl::range<2>{1, wg_size}},
        [=](sycl::nd_item<2> it) [[intel::reqd_sub_group_size(16)]] {
          const size_t r = it.get_global_id(0);
          const size_t c = it.get_global_id(1);

          *(vec + it.get_global_linear_id()) = *(vec_scratch + r * n + c);
        });
  });
}

sycl::event six_step_ifft(sycl::queue &q, ff_p254_t *vec,
                          ff_p254_t *vec_scratch, ff_p254_t *omega_dim_inv,
                          ff_p254_t *omega_n1_inv, ff_p254_t *omega_n2_inv,
                          ff_p254_t *omega_domain_size_inv, const uint64_t dim,
                          const uint64_t wg_size,
                          std::vector<sycl::event> evts) {
  assert((dim & (dim - 1ul)) == 0);

  uint64_t log_2_dim = (uint64_t)sycl::log2((float)dim);
  uint64_t n1 = 1 << (log_2_dim / 2);
  uint64_t n2 = dim / n1;
  uint64_t n = sycl::max(n1, n2);
  uint64_t log_2_n1 = (uint64_t)sycl::log2((float)n1);
  uint64_t log_2_n2 = (uint64_t)sycl::log2((float)n2);

  assert(n1 == n2 || n2 == 2 * n1);
  assert(log_2_dim > 0 && log_2_dim <= TWO_ADICITY_);

  // compute inverse of i-th root of unity, where i = {dim, n1, n2}
  sycl::event evt_0 = q.submit([&](sycl::handler &h) {
    h.depends_on(evts);
    h.single_task([=]() {
      *omega_dim_inv = static_cast<ff_p254_t>(
          cbn::mod_inv(get_root_of_unity(log_2_dim).data, mod_p254_bn));
      *omega_n1_inv = static_cast<ff_p254_t>(
          cbn::mod_inv(get_root_of_unity(log_2_n1).data, mod_p254_bn));
      *omega_n2_inv = static_cast<ff_p254_t>(
          cbn::mod_inv(get_root_of_unity(log_2_n2).data, mod_p254_bn));
      *omega_domain_size_inv = static_cast<ff_p254_t>(
          cbn::mod_inv(ff_p254_t(dim).data, mod_p254_bn));
    });
  });

  // Step 1: Transpose Matrix
  sycl::event evt_1 = matrix_transposed_initialise(q, vec, vec_scratch, n2, n1,
                                                   n, wg_size, evts);

  // Step 2: n2-many parallel n1-point Cooley-Tukey style IFFT
  sycl::event evt_2 = row_wise_transform(q, vec_scratch, omega_n1_inv, n2, n1,
                                         n, wg_size, {evt_0, evt_1});

  // Step 3: Multiply by twiddle factors
  sycl::event evt_3 = twiddle_multiplication(q, vec_scratch, omega_dim_inv, n2,
                                             n1, n, wg_size, {evt_2});

  // Step 4: Transpose Matrix
  sycl::event evt_4 = matrix_transpose(q, vec_scratch, n, {evt_3});

  // Step 5: n1-many parallel n2-point Cooley-Tukey IFFT
  sycl::event evt_5 = row_wise_transform(q, vec_scratch, omega_n2_inv, n1, n2,
                                         n, wg_size, {evt_4});

  // Step 6: Transpose Matrix
  sycl::event evt_6 = matrix_transpose(q, vec_scratch, n, {evt_5});

  // copy result back to source vector, while
  // also multiplying by inverse of domain size
  return q.submit([&](sycl::handler &h) {
    h.depends_on({evt_6});
    h.parallel_for<class kernelIFFTCopyBack>(
        sycl::nd_range<2>{sycl::range<2>{n2, n1}, sycl::range<2>{1, wg_size}},
        [=](sycl::nd_item<2> it) [[intel::reqd_sub_group_size(16)]] {
          const size_t r = it.get_global_id(0);
          const size_t c = it.get_global_id(1);

          *(vec + it.get_global_linear_id()) =
              *omega_domain_size_inv * *(vec_scratch + r * n + c);
        });
  });
}
