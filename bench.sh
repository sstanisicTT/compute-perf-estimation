#!/bin/bash

benchmarks=(
    "pytest tests/ttnn/nightly/unit_tests/operations/conv/test_conv2d.py -k test_conv_features_multi_device"
    "pytest tests/ttnn/unit_tests/operations/matmul/test_matmul.py -k test_matmul_in1_dram_sharded_tiny_tile"
    "pytest tests/ttnn/unit_tests/operations/matmul/test_matmul.py -k test_padded_1d_matmul"
    "pytest tests/ttnn/unit_tests/operations/eltwise/test_eltwise_selu.py -k test_eltwise_selu"
    "pytest tests/ttnn/unit_tests/operations/eltwise/test_binary_add_uint32.py -k test_binary_add_uint32_bcast"
    "pytest tests/ttnn/unit_tests/operations/conv/test_conv3d.py -k test_conv3d_sweep_shapes"
    "pytest tests/ttnn/unit_tests/operations/matmul/test_matmul.py -k test_sharded_matmul"
    "pytest tests/ttnn/unit_tests/operations/pool/test_maxpool2d.py"
)


BRANCH=$1   #baseline, profiler, counter

for i in {3..30}; do
    REPORT_DIR="../runs/$BRANCH/$i"
    mkdir -p "$REPORT_DIR"

    cd tt-metal
    git switch archive/compute-perf-$BRANCH

    for command in "${benchmarks[@]}"; do
        python3 -m tracy -o "$REPORT_DIR" -v -r -p -m "$command"
    done
done