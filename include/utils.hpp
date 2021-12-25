#pragma once
#include <ntt.hpp>
#include <random>
#include <chrono>

typedef std::chrono::_V2::system_clock::time_point tp;

void prepare_random_vector(ff_p254_t *vec, uint64_t n);
