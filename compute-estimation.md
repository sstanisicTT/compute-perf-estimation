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

Because the threads can be waiting at the same time.

## How to measure?

- `KERNEL_LENGTH` - Using a profiler zone around the kernel (already exists)
- `CB_WAIT_FRONT_TIME, CB_RESERVE_BACK_TIME` - More problematic

## Why problematic?
While the zone that captures the kernel length would only be entered once, if we put a zone inside
`llk_wait` (underlying call for `cb_wait_front`), the zone would be entered once for each
`cb_wait_front` call. This means that for kernels that call wait and reserve very often would
accumulate the overhead of entering a zone quickly. This would then impact the measurement of
`KERNEL_LENGTH`

Measurements for proof:

Profiler on average slows down the kernel by 5%, But more importantly it can impact the perf by up
to 10% specific kernels:
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
the number of cycles per loop by looking at the dissasembly and manually counting the cycles or
by comparing the output of za profiler zone to the output of the counter as separate runs.

With our POC implementation we calculated approximately 10 cycles per loop, but this might change
as the implementation changes.

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