# Game of Life: GPU Porting & Performance Analysis (HIP & OpenMP)

> **Related Repository:** For the CPU profiling (`gprof`), cache-locality loop optimisation, and OpenMP multithreading study on the same simulation, see [`gol-cpu-openmp`](https://github.com/md-173/gol-cpu-openmp).

Porting and benchmarking a serial C 2D cellular grid simulation (Conway's Game of Life) on a single **AMD Instinct MI250X** GPU (`gfx90a`) on the Pawsey **Setonix** supercomputer across two GPU programming models:

1. **HIP (C++):** Explicit device memory management, manual 2D thread-block mapping, `hipEvent_t` kernel timing, and defensive API error checking.
2. **OpenMP Target Offloading (C):** Minimal directive-based GPU offloading (three `#pragma` directives) designed for compiler-managed execution and portability.

---

## Key Findings

* **HIP peak speedup:** **111.92x** over the serial CPU baseline on a **30,000 × 30,000** grid (900 million cells, 100 iterations), with kernel-only execution ($t_k$) rising from **0.18%** at $10^2$ to **63.34%** at $30,000^2$.
* **OpenMP peak speedup:** **74.01x** on a **5,000 × 5,000** grid, outperforming the CPU across every tested grid size due to minimal runtime setup overhead (**~3 ms** vs. **~350 ms** for HIP).
* **Performance crossover:** OpenMP is faster from $10^2$ up to $10,000^2$ (`0.565 s` vs. `0.628 s`). **HIP overtakes OpenMP at $20,000^2$ (1.7x faster)** and widens the lead to **2.0x faster at $30,000^2$** (`2.799 s` vs. `5.604 s`), where OpenMP's compiler-managed thread mapping degrades (~16% increase in per-cell execution time, from `5.36 ns` at $5,000^2$ to `6.23 ns` at $30,000^2$).

---

## Implementation Highlights

### 1. Explicit GPU Port: HIP (C++)
* **Toolchain:** `rocm/6.4.1`, compiled via `hipcc --offload-arch=gfx90a`.
* **Thread-block geometry:** Fixed **16 × 16** blocks (256 threads = **4 wavefronts of 64 threads** on MI250X), launched with ceiling-rounded grid dimensions `((m + 15) / 16, (n + 15) / 16)` and an `if (i < n && j < m)` bounds guard for arbitrary non-square grids.
* **Minimised PCIe transfers:** Allocated two persistent device buffers (`dev_grid`, `dev_updated_grid`) via `hipMalloc`, performing only two `hipMemcpy` transfers (initial state in, final state out) and swapping device pointers on the host between timesteps.
* **Error handling & profiling:** Wrapped all HIP API calls in a `HIP_CHECK_ERROR` macro printing file/line diagnostics, checked asynchronous launch errors with `hipGetLastError()`, and isolated kernel execution time using `hipEvent_t` pairs (`hipEventRecord` / `hipEventElapsedTime`).

### 2. Directive-Based GPU Port: OpenMP Target Offloading (C)
* **Toolchain:** `rocm/6.3.0` + `craype-accel-amd-gfx90a`, compiled with `-fopenmp --offload-arch=gfx90a`.
* **Persistent device memory:** Used unstructured data directives in `main()` — `#pragma omp target enter data map(to: grid[0:n*m]) map(alloc: updated_grid[0:n*m])` and a matching `target exit data` with `map(from:)` / `map(delete:)` — avoiding unnecessary transfer of uninitialised working buffers. Host pointer swaps work without remapping because OpenMP tracks mappings by pointer address.
* **Compute offloading:** Parallelised the nested update loop with a single combined directive: `#pragma omp target teams distribute parallel for collapse(2) private(n_i, n_j)`, flattening the 2D iteration space for load balancing and isolating per-thread neighbour coordinate arrays to prevent race conditions.

---

## Correctness Verification

* **HIP:** Compiled the unmodified serial CPU solver directly into the GPU binary (`#define INCLUDE_CPU_VERSION`) and validated final grid buffers via `memcmp`, confirming **byte-identical output across all 11 grid configurations** ($10^2$ through $30,000^2$, including $1,000 \times 10,000$).
* **OpenMP:** Compared final state outputs against the serial CPU executable using `diff`, confirming **byte-identical output** on a $1,000 \times 1,000$ grid over 100 iterations.

---

## Performance Benchmarks (100 Iterations, Setonix MI250X)

*(Tip: Export Figures 1–3 from your report into `docs/` and uncomment the image links below.)*

<!--
![CPU vs GPU execution time](docs/cpu_vs_gpu_time.png)
![HIP vs OpenMP runtime comparison](docs/hip_vs_openmp_runtime.png)
![HIP vs OpenMP speedup comparison](docs/hip_vs_openmp_speedup.png)
-->

### Direct Runtime Comparison: HIP vs. OpenMP ($T_p$)

| Grid ($n \times m$) | HIP $T_p$ (s) | OpenMP $T_p$ (s) | Faster Model |
|---|---:|---:|---|
| 10 × 10 | 0.348 | 0.003 | OpenMP (116x faster) |
| 1,000 × 1,000 | 0.358 | 0.008 | OpenMP (47x faster) |
| 5,000 × 5,000 | 0.428 | 0.134 | OpenMP (3.2x faster) |
| 10,000 × 10,000 | 0.628 | 0.565 | OpenMP (1.1x faster) |
| 20,000 × 20,000 | 1.456 | 2.414 | **HIP (1.7x faster)** |
| 30,000 × 30,000 | 2.799 | 5.604 | **HIP (2.0x faster)** |

### HIP Full Scaling Results

| Grid ($n \times m$) | Serial CPU $T_s$ (s) | GPU Total $T_p$ (s) | Speedup ($T_s / T_p$) | Kernel % ($t_k$) |
|---|---:|---:|---:|---:|
| 10 × 10 | 0.000028 | 0.348 | 0.0001x | 0.18% |
| 100 × 100 | 0.00358 | 0.358 | 0.01x | 0.18% |
| 500 × 500 | 0.0865 | 0.357 | 0.242x | 0.28% |
| 1,000 × 1,000 | 0.347 | 0.358 | 0.969x | 0.66% |
| 2,500 × 2,500 | 2.16 | 0.376 | 5.75x | 3.30% |
| 1,000 × 10,000 | 3.461 | 0.391 | 8.84x | 5.16% |
| 5,000 × 5,000 | 8.635 | 0.428 | 20.18x | 11.29% |
| 7,500 × 7,500 | 19.48 | 0.512 | 38.04x | 21.74% |
| 10,000 × 10,000 | 34.65 | 0.628 | 55.16x | 31.25% |
| 20,000 × 20,000 | 138.6 | 1.456 | 95.20x | 54.95% |
| 30,000 × 30,000 | 313.2 | 2.799 | **111.92x** | **63.34%** |

### OpenMP Target Offloading Full Scaling Results

| Grid ($n \times m$) | Serial CPU $T_s$ (s) | GPU Total $T_p$ (s) | Speedup ($T_s / T_p$) |
|---|---:|---:|---:|
| 10 × 10 | 0.0049 | 0.00259 | 1.89x |
| 100 × 100 | 0.006 | 0.00307 | 1.96x |
| 500 × 500 | 0.102 | 0.00423 | 24.10x |
| 1,000 × 1,000 | 0.400 | 0.00769 | 52.06x |
| 2,500 × 2,500 | 2.495 | 0.0351 | 71.02x |
| 1,000 × 10,000 | 3.963 | 0.0536 | 73.94x |
| 5,000 × 5,000 | 9.905 | 0.134 | **74.01x** |
| 7,500 × 7,500 | 22.33 | 0.305 | 73.12x |
| 10,000 × 10,000 | 39.71 | 0.565 | 70.27x |
| 20,000 × 20,000 | 159.0 | 2.414 | 65.85x |
| 30,000 × 30,000 | 358.2 | 5.604 | 63.92x |

---

## Build & Run (Pawsey Setonix)

### HIP (C++)
```bash
module load rocm/6.4.1
cd ex1-gol-hip
make

# Usage: ./bin/game_of_life_hip <rows> <cols> [steps]
./bin/game_of_life_hip 10000 10000 100
```

### OpenMP Target Offloading (C)
```bash
module load rocm/6.3.0
module load craype-accel-amd-gfx90a
cd ex2-gol-gpu-directives/openmp
make

# Usage: ./bin/<executable> <rows> <cols> [steps]
./bin/02_gol_gpu_openmp 5000 5000 100
```

---

## Repository Structure

```text
gol-gpu-hip-openmp/
├── ex1-gol-hip/                  # HIP (C++) GPU port, serial CPU baseline, Makefile
│   ├── src/
│   │   ├── game_of_life.c        # Serial reference implementation
│   │   └── game_of_life_hip.cpp  # HIP GPU kernel, timing, and memcmp verification
│   └── Makefile
├── ex2-gol-gpu-directives/       # OpenMP target offloading C port & Makefile
│   └── openmp/
│       ├── src/
│       └── Makefile
├── docs/                         # Exported scaling & comparison plots (PNG)
└── LICENSE
```
