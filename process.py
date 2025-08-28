#!/usr/bin/env python3

import os
import glob
import pandas as pd
from pathlib import Path
import argparse

def device_profiler_log_files(base_dir):
    files = []
    # Now, base_dir contains subdirectories 0, 1, ..., N, each with a "reports" subdir
    for subdir in os.listdir(base_dir):
        subdir_path = os.path.join(base_dir, subdir)
        reports_dir = os.path.join(subdir_path, "reports")
        print(f"Processing {reports_dir}")
        if os.path.isdir(reports_dir):
            for root, _, _ in os.walk(reports_dir):
                files.extend(glob.glob(os.path.join(root, "profile_log_device.csv")))
    return sorted(files)

def calculate_kernel_length(group):
    """Calculate KERNEL_LENGTH by finding ZONE_START and ZONE_END pairs"""
    zone_start = group[(group['zone_name'] == 'TRISC-KERNEL') & (group['type'] == 'ZONE_START')]
    zone_end = group[(group['zone_name'] == 'TRISC-KERNEL') & (group['type'] == 'ZONE_END')]
    

    if len(zone_start) == 0 or len(zone_end) == 0:
        print(f"Warning: {group} has no ZONE_START or ZONE_END")
        return 0

    if len(zone_start) > 1 or len(zone_end) > 1:
        print(f"Warning: {group} has multiple ZONE_START or ZONE_END")
        return 0

    # Get the latest start and end times for this zone
    start_time = zone_start['time_cycles'].max()
    end_time = zone_end['time_cycles'].max()
    return end_time - start_time


def extract_cb_metrics(df):
    """Extract CB-COMPUTE metrics if they exist"""
    cb_wait_front = df[df['zone_name'] == 'CB-COMPUTE-WAIT-FRONT']
    cb_reserve_back = df[df['zone_name'] == 'CB-COMPUTE-RESERVE-BACK']
    
    metrics = {}
    
    if not cb_wait_front.empty:
        # Use the 'data' column (column 7) for CB timing information
        metrics['CB_WAIT_FRONT'] = cb_wait_front['data'].sum()
    
    if not cb_reserve_back.empty:
        # Use the 'data' column (column 7) for CB timing information  
        metrics['CB_RESERVE_BACK'] = cb_reserve_back['data'].sum()
    
    return metrics

def transform_profile_file(input_file, output_file):
    """Transform a single profile_log_device.csv file"""
    try:
        # Read CSV, skipping the first line with ARCH info
        df = pd.read_csv(input_file, skiprows=1, header=0)
        
        # Rename columns based on the CSV structure we analyzed
        expected_columns = [
            'pcie', 'core_x', 'core_y', 'risc_type', 'timer_id', 
            'time_cycles', 'data', 'run_host_id', 'zone_name', 'type', 
            'source_line', 'source_file', 'meta_data'
        ]
        
        if len(df.columns) >= len(expected_columns):
            df.columns = expected_columns[:len(df.columns)]
        else:
            print(f"Warning: {input_file} has fewer columns than expected")
            return False
        
        # Filter out BRISC and NCRISC processors - keep only TRISC
        filter_risc_type = df['risc_type'].str.contains('TRISC', na=False)
        df = df[filter_risc_type]

        # Group by core, risc_type, and run_host_id to calculate metrics per processor
        grouped = df.groupby(['pcie', 'core_x', 'core_y', 'risc_type', 'run_host_id'])

        # Remove groups that don't contain the TRISC-KERNEL zone
        def group_contains_trisc_kernel(group):
            return (group['zone_name'] == 'TRISC-KERNEL').any()

        # Filter the grouped object to only include groups with TRISC-KERNEL zone
        grouped = {k: g for k, g in grouped if group_contains_trisc_kernel(g)}


        result_df = pd.DataFrame([
            {
                'pcie': pcie,
                'core_x': core_x,
                'core_y': core_y,
                'risc_type': risc_type,
                'host_id': run_host_id,
                'KERNEL_LENGTH': calculate_kernel_length(group),
                **extract_cb_metrics(group)
            }
            for (pcie, core_x, core_y, risc_type, run_host_id), group in grouped.items()
        ])
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Save to CSV
        result_df.to_csv(output_file, index=False)
        print(f"Transformed: {input_file} -> {output_file}")
        return True
    
            
    except Exception as e:
        print(f"Error processing {input_file}: {e}")
        return False

def transform_directory(input_dir, output_dir):
    """Transform all profile_log_device.csv files maintaining directory structure"""
    files = device_profiler_log_files(input_dir)
    
    
    success_count = 0
    for input_file in files:
        rel_path = os.path.relpath(input_file, input_dir)
        output_file = os.path.join(output_dir, rel_path)
        
        if transform_profile_file(input_file, output_file):
            success_count += 1
    
    print(f"Successfully transformed {success_count}/{len(files)} files")

def main():
    parser = argparse.ArgumentParser(description='Transform profile_log_device.csv files')
    parser.add_argument('--input', '-i', default='runs', 
                      help='Input directory containing runs (default: runs)')
    parser.add_argument('--output', '-o', default='processed', 
                      help='Output directory for processed files (default: processed)')
    
    args = parser.parse_args()
    
    input_dir = args.input
    output_dir = args.output
    
    if not os.path.exists(input_dir):
        print(f"Error: Input directory {input_dir} does not exist")
        return
    
    print(f"Transforming profile files from {input_dir} to {output_dir}")
    
    subdirs = os.listdir(input_dir)

    # Process each subdirectory (baseline, counter, profiler)
    for subdir in subdirs:
        input_subdir = os.path.join(input_dir, subdir)
        output_subdir = os.path.join(output_dir, subdir)
        
        print(f"\nProcessing {subdir} directory...")
        transform_directory(input_subdir, output_subdir)
        
if __name__ == "__main__":
    main()
