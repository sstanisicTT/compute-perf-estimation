# Isolating compute performance when measuring compute

We currently don't have a way to easily isolate the performance of compute from datamovement.
The approach that is currently used is to manually comment out the NOC operations in the
relevant reader kernel and measure the KERNEL_LENGTH after changing the reader. This is a
tedious and error-prone process. We want to propose an alternative way to isolate compute.

## The Idea

Start from the assumption:

```
COMPUTE_TIME = KERNEL_LENGTH - BLOCKED_TIME
```

- KERNEL_LENGTH is time from kernel starting to execute on Unpacker, and finishing on the Packer.
- BLOCKED_TIME is time that the compute kernel is waiting for Data Movement

From this we can make the estimation:

```
COMPUTE_TIME = KERNEL_LENGTH - CB_WAIT_FRONT_TIME
```

or

```
COMPUTE_TIME = KERNEL_LENGTH - CB_RESERVE_BACK
```

But not

```
COMPUTE_TIME = KERNEL_LENGTH - (CB_WAIT_FRONT_TIME + CB_RESERVE_BACK)
```

Because the threads can be waiting at the same time. Specifically, the unpacker thread waits on `cb_wait_front` while the packer thread blocks on `cb_reserve_back`, and these waits can occur concurrently. This concurrent waiting means that simply adding the two wait times would overestimate the blocked time, leading to an underestimate of the actual compute time.


## How to measure?

- `KERNEL_LENGTH` - Using a profiler zone around the kernel (already exists)
- `CB_WAIT_FRONT_TIME, CB_RESERVE_BACK_TIME` - More problematic

## Why problematic?
While the zone that captures the kernel length would only be entered once, if we put a zone inside
`llk_wait` (underlying call for `cb_wait_front`), the zone would be entered once for each
`cb_wait_front` call. This means that for kernels that call wait and reserve very often would
accumulate the overhead of entering a zone quickly. This would then impact the measurement of
`KERNEL_LENGTH`

## Test Benchmark Setup

The overhead measurements are based on a comprehensive benchmark suite consisting of 8 different test cases:

```
pytest tests/ttnn/nightly/unit_tests/operations/conv/test_conv2d.py -k test_conv_features_multi_device
pytest tests/ttnn/unit_tests/operations/matmul/test_matmul.py -k test_matmul_in1_dram_sharded_tiny_tile
pytest tests/ttnn/unit_tests/operations/matmul/test_matmul.py -k test_padded_1d_matmul
pytest tests/ttnn/unit_tests/operations/eltwise/test_eltwise_selu.py -k test_eltwise_selu
pytest tests/ttnn/unit_tests/operations/eltwise/test_binary_add_uint32.py -k test_binary_add_uint32_bcast
pytest tests/ttnn/unit_tests/operations/conv/test_conv3d.py -k test_conv3d_sweep_shapes
pytest tests/ttnn/unit_tests/operations/matmul/test_matmul.py -k test_sharded_matmul
pytest tests/ttnn/unit_tests/operations/pool/test_maxpool2d.py
```

Each test is executed 30 times (runs 0-30) to ensure statistical significance. The overhead is
measured by comparing kernel length performance between the baseline implementation and
instrumented versions (profiler vs counter approaches).

## Measurements for proof:

The analysis follows a three-stage pipeline:
1. **Processing**: Raw Tracy profiler logs are processed to extract `KERNEL_LENGTH` values (time from kernel start on Unpacker to finish on Packer) for each core/run combination
2. **Unification**: All runs for each implementation (baseline/counter/profiler) are combined and statistics calculated row-by-row across the 30 repetitions
3. **Comparison**: Performance impact is calculated as percentage slowdown: `(instrumented_time - baseline_time) / baseline_time * 100`

### Metrics Explained:
- **Average Slowdown**: Mean percentage increase in `KERNEL_LENGTH` across all test cases compared to baseline
- **Std Dev**: Standard deviation of the slowdown from the averege slowdown
- **Median Slowdown**: Median percentage increase in `KERNEL_LENGTH` across all test cases compared to baseline

Over the whole benchmark, profiler on average slows down the kernel by 5%, But more importantly it
can impact the perf by up to 10% for specific kernels:
```
PROFILER vs BASELINE:
  KERNEL_LENGTH Performance Impact:
    Average Slowdown: 4.880%
    Std Dev:          8.080%
    Median Slowdown:  1.304%

MAXPOOL (Outlier):
PROFILER vs BASELINE:
  KERNEL_LENGTH Performance Impact:
    Average Slowdown: 8.514%
    Std Dev:          10.332%
    Median Slowdown:  3.100%
```

This is because the `DeviceZoneScopedSumN` adds 10-15 cycles per entry:
- Writing the zone id to local memory
- Reading start WALL_CLOCK
- Reading the current sum from local memory
- Reading end WALL_CLOCK
- Writing the updated sum to local memory

## Proposed solution

```
inline void llk_wait_for_free_tiles(const std::int32_t operand, const std::int32_t num_tiles) {
    // .....

    uint32_t counter = LOAD_COUNTER();  // Load counter from local memory (1-2 cycles)

    do {
        counter++;                      // Increment  counter (1 cycle)

        std::uint16_t tiles_acked = (std::uint16_t)reg_read((std::uint32_t)tiles_acked_ptr);
        std::uint32_t free_tiles_wrap = get_local_cb_interface(output).fifo_num_pages - (tiles_received - tiles_acked);
        free_tiles = (std::int32_t)free_tiles_wrap;
    } while (free_tiles < num_tiles);

    STORE_COUNTER(--counter);           // Store counter (1 cycle)
}

```


Insteadx of trying to calculate the exact number of cycles, just count the number of extra
iterations spent inside of the loop that waits for Data Movement.

This would simplify the algorithm to the following:
- Reading the current counter from local memory
- Locally (in register) increasing the counter for each iteration of the loop
- Writing the updated counter to local memory

From our proof of concept experiment this reduces the overhead to 5 cycles per call

Kernel length impact:

```
Proposed solution
COUNTER vs BASELINE:
  KERNEL_LENGTH Performance Impact:
    Average Slowdown: 0.753%
    Std Dev:          2.076%
    Median Slowdown:  0.309%

DeviceZoneScopedSumN approach
PROFILER vs BASELINE:
  KERNEL_LENGTH Performance Impact:
    Average Slowdown: 4.880%
    Std Dev:          8.080%
    Median Slowdown:  1.304%
```

From the counter value we can calculate the number of cycles spent inside of the loop. We can find
the number of cycles per loop by looking at the disassembly and manually counting the cycles or
by comparing the output of a profiler zone to the output of the counter as separate runs.

### Cycle Calculation Methods

**Method 1: Disassembly Analysis**
```
lw   a3,0(t3)
lhu  s5,24(a1)       # outside of loop

.L15:  ; Loop start
lw   a5,16(a7)       # 1 cycle here, 2 cycles when used in next instruction (data hazard)
mv   s4,a3           # 1 cycle
sub  a5,a5,s5        # 1 cycle
slli a5,a5,0x10      # 1 cycle
srli a5,a5,0x10      # 1 cycle
addi a3,a3,1         # 1 cycle
bltu a5,a2,5e68      # 1 cycle (2 if the branch is misspredicted)

sw   s4,0(t3)        # outside of loop
```

Total per loop

**Method 2: Profiler Calibration**
Run the same kernel with both profiler zones and counters enabled separately:

```
inline void llk_wait_for_free_tiles(const std::int32_t operand, const std::int32_t num_tiles) {
    // .....

    uint32_t counter = LOAD_COUNTER();  // Load counter from local memory (1-2 cycles)

    do {
        counter++;                      // Increment  counter (1 cycle)

        std::uint16_t tiles_acked = (std::uint16_t)reg_read((std::uint32_t)tiles_acked_ptr);
        std::uint32_t free_tiles_wrap = get_local_cb_interface(output).fifo_num_pages - (tiles_received - tiles_acked);
        free_tiles = (std::int32_t)free_tiles_wrap;
    } while (free_tiles < num_tiles);

    STORE_COUNTER(--counter);           // Store counter (1 cycle)
}

```
inline void llk_wait_for_free_tiles(const std::int32_t operand, const std::int32_t num_tiles) {
    // .....

    {
        DeviceZoneScopedSumN("ZONE")
        do {
            std::uint16_t tiles_acked = (std::uint16_t)reg_read((std::uint32_t)tiles_acked_ptr);
            std::uint32_t free_tiles_wrap = get_local_cb_interface(output).fifo_num_pages - (tiles_received - tiles_acked);
            free_tiles = (std::int32_t)free_tiles_wrap;
        } while (free_tiles < num_tiles);
    }

    // ...
}
```


### Geek Numbers

Numbers are measured over 8 different tests, repeated 30 times. The slowdown is calculated for
each measurement compared to baseline and averaged.

Wormhole overhead measurements:
```
COUNTER vs BASELINE:
  KERNEL_LENGTH Performance Impact:
    Average Slowdown: 0.753%
    Std Dev:          2.076%
    Median Slowdown:  0.309%

PROFILER vs BASELINE:
  KERNEL_LENGTH Performance Impact:
    Average Slowdown: 4.880%
    Std Dev:          8.080%
    Median Slowdown:  1.304%
```

Wormhole standard deviation:
```
BASELINE
STD Percentage statistics:
    Mean: 1.3934%
    Median: 0.4234%
    Min: 0.0000%
    Max: 37.5139%
    25th percentile: 0.1064%
    75th percentile: 2.0198%

COUNTER
STD Percentage statistics:
    Mean: 1.4142%
    Median: 0.4448%
    Min: 0.0022%
    Max: 38.2072%
    25th percentile: 0.1065%
    75th percentile: 2.0401%

PROFILER
STD Percentage statistics:
    Mean: 1.3894%
    Median: 0.4157%
    Min: 0.0028%
    Max: 37.3193%
    25th percentile: 0.1095%
    75th percentile: 2.0290%
```

Maxpool Wormhole:
```
COUNTER vs BASELINE:
KERNEL_LENGTH Performance Impact:
    Average Slowdown: 1.214%
    Std Dev:          2.688%
    Median Slowdown:  0.390%

PROFILER vs BASELINE:
KERNEL_LENGTH Performance Impact:
    Average Slowdown: 8.514%
    Std Dev:          10.332%
    Median Slowdown:  3.100%
```

Blackhole:
```
COUNTER vs BASELINE:
KERNEL_LENGTH Performance Impact:
    Average Slowdown: 0.162%
    Std Dev:          1.302%
    Median Slowdown:  0.083%

PROFILER vs BASELINE:
KERNEL_LENGTH Performance Impact:
    Average Slowdown: 3.131%
    Std Dev:          4.929%
    Median Slowdown:  0.872%
```

Blackhole standard deviation:
```
BASELINE
STD Percentage statistics:
    Mean: 1.3128%
    Median: 0.3349%
    Min: 0.0004%
    Max: 48.9884%
    25th percentile: 0.0479%
    75th percentile: 1.6731%

COUNTER
STD Percentage statistics:
    Mean: 1.2996%
    Median: 0.3453%
    Min: 0.0004%
    Max: 49.6464%
    25th percentile: 0.0504%
    75th percentile: 1.6936%

PROFILER
STD Percentage statistics:
    Mean: 1.3267%
    Median: 0.4344%
    Min: 0.0004%
    Max: 56.4012%
    25th percentile: 0.0801%
    75th percentile: 1.6543%
```