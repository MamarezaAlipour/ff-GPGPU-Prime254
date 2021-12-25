## Benchmarking (I)NTT on 254-bit Prime Field

I had access to Nvidia Tesla V100 GPU, so I ran benchmark suite with CUDA backend.

> Note: Following numbers don't include time required to transfer input/ output in between host and device.

```bash
DO_RUN=benchmark make cuda && ./run

running on Tesla V100-SXM2-16GB
```

### NTT

```bash
Six-Step FFT (without data transfer cost)

  dimension		          total
    65536		         18.336 ms
   131072		         18.305 ms
   262144		         34.144 ms
   524288		         68.081 ms
  1048576		        135.881 ms
  2097152		        273.544 ms
  4194304		        559.814 ms
  8388608		        1154.44 ms
```

```bash
Six-Step FFT (with host -> device data transfer cost)

  dimension		          total
    65536		          9.401 ms
   131072		         17.686 ms
   262144		         33.732 ms
   524288		         67.518 ms
  1048576		        136.056 ms
  2097152		        279.123 ms
  4194304		         571.29 ms
  8388608		        1179.25 ms
```

```bash
Six-Step FFT (with device -> host data transfer cost)

  dimension		          total
    65536		          9.338 ms
   131072		         17.556 ms
   262144		         33.626 ms
   524288		         67.541 ms
  1048576		        135.421 ms
  2097152		        278.522 ms
  4194304		        569.943 ms
  8388608		           1176 ms
```

```bash
Six-Step FFT (with host <-> device data transfer cost)

  dimension		          total
    65536		          9.661 ms
   131072		         18.006 ms
   262144		         34.313 ms
   524288		         71.073 ms
  1048576		        140.953 ms
  2097152		         284.84 ms
  4194304		        581.876 ms
  8388608		        1198.76 ms
```

### INTT

```bash
Six-Step IFFT (without data transfer cost)

  dimension		          total
    65536		         10.034 ms
   131072		         18.154 ms
   262144		         34.154 ms
   524288		         68.342 ms
  1048576		        136.843 ms
  2097152		        279.982 ms
  4194304		        572.888 ms
  8388608		        1181.71 ms
```

```bash
Six-Step IFFT (with host -> device data transfer cost)

  dimension		          total
    65536		         10.163 ms
   131072		         18.754 ms
   262144		          35.01 ms
   524288		         69.904 ms
  1048576		        140.006 ms
  2097152		        290.209 ms
  4194304		        584.764 ms
  8388608		        1204.64 ms
```

```bash
Six-Step IFFT (with device -> host data transfer cost)

  dimension		          total
    65536		         10.142 ms
   131072		         18.551 ms
   262144		         34.803 ms
   524288		         69.593 ms
  1048576		        139.473 ms
  2097152		        285.581 ms
  4194304		        583.526 ms
  8388608		        1202.13 ms
```

```bash
Six-Step IFFT (with host <-> device data transfer cost)

  dimension		          total
    65536		         10.501 ms
   131072		         18.916 ms
   262144		         35.704 ms
   524288		         71.154 ms
  1048576		        142.749 ms
  2097152		          291.9 ms
  4194304		        596.034 ms
  8388608		           1229 ms
```
