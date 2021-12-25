# ff-GPGPU-Prime254
Accelerating Computation on 254-bit Prime Field

## Motivation

In recent times I got opportunity to work with people who're exploring Zero Knowledge Cryptography using S{N,T}ARKS and I was requested to explore possibilities of accelerating compute heavy algorithms, by offloading them to accelerators, specifically GPUs.

SYCL is a modern abstraction for writing data parallel programs, which caught my attention due to its ability of interacting with devices of heterogeneous nature, starting from CPU, GPU to FPGA, with multiple backends support such that OpenCL, CUDA, HIP etc..

In this repository, I keep following implementations, where elements are chosen from 254-bit prime field [`F(21888242871839275222246405745257275088548364400416034343698204186575808495617)`](https://github.com/PariKhaleghi/ff-GPGPU-Prime254/blob/master/include/ntt.hpp#L7-L11)

- (Inverse) Number Theoretic Transform

I've written test cases and benchmark suite, in SYCL/ DPC++, to better understand how beneficial is it to offload computation to accelerators.

## Prerequisite

- I'm using 

```bash
$ lsb_release -d

Description:	Ubuntu 20.04.3 LTS
```

- I've access to Nvidia Tesla V100 GPU

```bash
$ lspci | grep -i nvidia

00:1e.0 3D controller: NVIDIA Corporation GV100GL [Tesla V100 SXM2 16GB] (rev a1)
```

- I've installed CUDA toolkit by following [this document](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html).

- I've compiled Intel's SYCL implementation with CUDA support, while following [this document](https://intel.github.io/llvm-docs/GetStartedGuide.html#build-dpc-toolchain-with-support-for-nvidia-cuda)

- This is the compiler which I'll use for compiling host/ device code

```bash
$ clang++ --version

clang version 14.0.0 (https://github.com/intel/llvm 9ca7ceac3bfe24444209f56567ca50e51dd9e5cf)
Target: x86_64-unknown-linux-gnu
Thread model: posix
InstalledDir: /home/ubuntu/sycl_workspace/llvm/build/bin
```

- `make` and other system development toolchains are required for running this test cases and benchmark suite.

- As you probably already noticed, I'll be working on accelerating 254-bit prime field computation, I make use of [`ctbignum`](https://github.com/niekbouman/ctbignum) as arbitrary precision modular arithmetic library.

- Run following code snippet to download proper version of `ctbignum`

```bash
cd ~
git clone https://github.com/niekbouman/ctbignum

pushd ctbignum
git checkout e8fdb0d6f7d304fb1eed2029078d3f653c4f67db # **important**
sudo cp -r include/ctbignum /usr/local/include
popd
```

- `ctbignum` uses modern C++ features, so C++20 headers are required.

```bash
sudo apt-get install libstdc++-10-dev
```

## Usage

For running test cases

```bash
DO_RUN=test make cuda
./run
```

For running benchmark suite

```bash
DO_RUN=benchmark make cuda
./run
```

> I suggest you take a look at Makefile for more build recipes.

## Benchmarks

- [(I)NTT on GPU ( CUDA backend )](benchmarks/ntt.md)
