#!/usr/bin/env python3
"""
Script to analyze data in the unified folder and calculate row-by-row statistics.
For each row position, calculates averages and variance for KERNEL_LENGTH between runs.
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

def analyze_category(category_path, category_name):
    """Analyze all runs for a given category and calculate row-by-row statistics."""
    print(f"Analyzing category: {category_name}")
    
    # Find all run directories (0, 1, 2, 3, 4)
    run_dirs = sorted([d for d in os.listdir(category_path) if d.isdigit()])
    
    if not run_dirs:
        print(f"No run directories found for {category_name}")
        return None
    
    # Load data from all runs
    run_dataframes = {}
    
    for run_dir in run_dirs:
        csv_file = os.path.join(category_path, run_dir, f"{category_name}.csv")
        
        if not os.path.exists(csv_file):
            print(f"Warning: {csv_file} not found, skipping...")
            continue
        
        print(f"  Loading run {run_dir}...")
        try:
            df = pd.read_csv(csv_file)
            run_dataframes[int(run_dir)] = df
            print(f"    Loaded {len(df)} records")
        except Exception as e:
            print(f"  Error reading {csv_file}: {e}")
            continue
    
    if not run_dataframes:
        print(f"No valid data found for {category_name}")
        return None
    
    # Verify all dataframes have the same length and structure
    first_run = list(run_dataframes.keys())[0]
    reference_df = run_dataframes[first_run]
    num_rows = len(reference_df)
    
    print(f"  Verifying data consistency across runs...")
    for run_id, df in run_dataframes.items():
        if len(df) != num_rows:
            print(f"  Warning: Run {run_id} has {len(df)} rows, expected {num_rows}")
            return None
    
    print(f"  All runs have {num_rows} rows - proceeding with analysis")
    
    # Start with basic identifying columns from reference dataframe
    result_df = reference_df[['run_id', 'host_id', 'pcie', 'core_x', 'core_y', 'risc_type']].copy()
    
    # Calculate row-by-row statistics for KERNEL_LENGTH
    if 'KERNEL_LENGTH' in reference_df.columns:
        print(f"  Calculating KERNEL_LENGTH statistics...")
        
        # Collect KERNEL_LENGTH values for each row across all runs
        kernel_values_matrix = []
        for run_id in sorted(run_dataframes.keys()):
            kernel_values_matrix.append(run_dataframes[run_id]['KERNEL_LENGTH'].values)
        
        # Convert to numpy array for easier calculation (rows x runs)
        kernel_matrix = np.array(kernel_values_matrix).T  # Transpose so each row represents a data point
        
        # Calculate mean and standard deviation for each row
        kernel_means = np.mean(kernel_matrix, axis=1)
        kernel_std_devs = np.std(kernel_matrix, axis=1, ddof=1)  # Sample standard deviation
        
        # Add to result dataframe
        result_df['KERNEL_LENGTH_AVG'] = kernel_means
        result_df['KERNEL_LENGTH_STD'] = kernel_std_devs
    

    
    print(f"  Analysis complete - {len(result_df)} rows with statistics")
    return result_df

def save_results(result_df, category_name, output_dir):
    """Save the results with statistics to CSV file."""
    if result_df is None:
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save to CSV
    output_file = os.path.join(output_dir, f"{category_name}.csv")
    result_df.to_csv(output_file, index=False)
    print(f"Results saved to: {output_file}")
    
    # Print summary statistics
    print(f"  Summary for {category_name}:")
    if 'KERNEL_LENGTH_AVG' in result_df.columns:
        overall_kernel_mean = result_df['KERNEL_LENGTH_AVG'].mean()
        overall_kernel_std = result_df['KERNEL_LENGTH_STD'].mean()
        print(f"    KERNEL_LENGTH - Overall avg of averages: {overall_kernel_mean:.2f}")
        print(f"    KERNEL_LENGTH - Overall avg of std devs: {overall_kernel_std:.2f}")

def main():
    """Main function to analyze all categories."""
    # Define paths
    unified_dir = "/Users/sstanisic/code/counter-data/unified"
    statistics_dir = "/Users/sstanisic/code/counter-data/statistics"
    
    # Categories to analyze
    categories = ['baseline', 'counter', 'profiler']
    
    print("Starting row-by-row data analysis...")
    print(f"Input directory: {unified_dir}")
    print(f"Output directory: {statistics_dir}")
    print("-" * 50)
    
    # Analyze each category
    for category in categories:
        category_path = os.path.join(unified_dir, category)
        
        if not os.path.exists(category_path):
            print(f"Category directory not found: {category_path}")
            continue
        
        # Analyze the category (returns dataframe with statistics)
        result_df = analyze_category(category_path, category)
        
        if result_df is not None:
            # Save results
            save_results(result_df, category, statistics_dir)
        
        print("-" * 50)
    
    print("Analysis complete!")

if __name__ == "__main__":
    main()
