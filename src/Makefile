CXX = clang++
CXXFLAGS = -std=c++20 -Wall
SYCLFLAGS = -fsycl
SYCLCUDAFLAGS = -fsycl-targets=nvptx64-nvidia-cuda
INCLUDES = -I./include
PROG = run

# expects user to set DO_RUN variable to either of
# {test, benchmark}
#
# If `DO_RUN=test make` is invoked, compiled binary will
# run test cases only
#
# On other hand, if `DO_RUN=benchmark make` is invoked,
# compiled binary will only run benchmark suite
#
# if nothing is set, none is used, which results into
# compiling binary which neither runs test cases nor runs
# benchmark suite !
DFLAGS = -D$(shell echo $(or $(DO_RUN),nothing) | tr a-z A-Z)

$(PROG): main.o test.o ntt.o utils.o bench_ntt.o
	$(CXX) $(SYCLFLAGS) $^ -o $@

bench_ntt.o: bench_ntt.cpp include/bench_ntt.hpp
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) $(INCLUDES) -c $<

utils.o: utils.cpp include/utils.hpp
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) $(INCLUDES) -c $<

ntt.o: ntt.cpp include/ntt.hpp
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) $(INCLUDES) -c $<

test.o: test/test.cpp include/test.hpp
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) $(INCLUDES) -c $<

main.o: main.cpp include/test.hpp include/bench_ntt.hpp
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) $(DFLAGS) $(INCLUDES) -c $<

aot_cpu:
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) $(DFLAGS) -c main.cpp -o main.o $(INCLUDES)
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) -c utils.cpp -o utils.o $(INCLUDES)
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) -c test/test.cpp -o test.o $(INCLUDES)
	@if lscpu | grep -q 'avx512'; then \
		echo "Using avx512"; \
		$(CXX) $(CXXFLAGS) $(SYCLFLAGS) $(INCLUDES) -fsycl-targets=spir64_x86_64-unknown-unknown-sycldevice -Xs "-march=avx512" bench_ntt.cpp ntt.cpp test.o utils.o main.o; \
	elif lscpu | grep -q 'avx2'; then \
		echo "Using avx2"; \
		$(CXX) $(CXXFLAGS) $(SYCLFLAGS) $(INCLUDES) -fsycl-targets=spir64_x86_64-unknown-unknown-sycldevice -Xs "-march=avx2" bench_ntt.cpp ntt.cpp test.o utils.o main.o; \
	elif lscpu | grep -q 'avx'; then \
		echo "Using avx"; \
		$(CXX) $(CXXFLAGS) $(SYCLFLAGS) $(INCLUDES) -fsycl-targets=spir64_x86_64-unknown-unknown-sycldevice -Xs "-march=avx" bench_ntt.cpp ntt.cpp test.o utils.o main.o; \
	elif lscpu | grep -q 'sse4.2'; then \
		echo "Using sse4.2"; \
		$(CXX) $(CXXFLAGS) $(SYCLFLAGS) $(INCLUDES) -fsycl-targets=spir64_x86_64-unknown-unknown-sycldevice -Xs "-march=sse4.2" bench_ntt.cpp ntt.cpp test.o utils.o main.o; \
	else \
		echo "Can't AOT compile using avx, avx2, avx512 or sse4.2"; \
	fi

aot_gpu:
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) $(DFLAGS) -c main.cpp -o main.o $(INCLUDES)
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) -c utils.cpp -o utils.o $(INCLUDES)
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) -c test/test.cpp -o test.o $(INCLUDES)
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) $(INCLUDES) -fsycl-targets=spir64_gen-unknown-unknown-sycldevice -Xs "-device 0x4905" bench_ntt.cpp ntt.cpp test.o utils.o main.o

clean:
	find . -name '*.o' -o -name 'a.out' -o -name 'run' -o -name '*.gch' | xargs rm -f

format:
	find . -name '*.cpp' -o -name '*.hpp' | xargs clang-format -i

cuda:
	# make sure you've built `clang++` with CUDA support
	# check https://intel.github.io/llvm-docs/GetStartedGuide.html#build-dpc-toolchain-with-support-for-nvidia-cuda
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) $(SYCLCUDAFLAGS) $(DFLAGS) -c main.cpp -o main.o $(INCLUDES)
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) $(SYCLCUDAFLAGS) -c utils.cpp -o utils.o $(INCLUDES)
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) $(SYCLCUDAFLAGS) -c test/test.cpp -o test.o $(INCLUDES)
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) $(SYCLCUDAFLAGS) -c bench_ntt.cpp -o bench_ntt.o $(INCLUDES)
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) $(SYCLCUDAFLAGS) -c ntt.cpp -o ntt.o $(INCLUDES)
	$(CXX) $(SYCLFLAGS) $(SYCLCUDAFLAGS) *.o -o $(PROG)
