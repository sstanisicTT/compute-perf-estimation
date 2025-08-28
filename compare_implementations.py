#!/usr/bin/env python3
"""
Script to compare counter and profiler implementations against baseline.
Calculates relative error for KERNEL_LENGTH mean and standard deviation.
"""

import pandas as pd
import numpy as np
import os

def calculate_percentage_slowdown(baseline_value, comparison_value):
    """Calculate percentage slowdown: (comparison - baseline) / baseline * 100
    Positive values indicate slowdown, negative values indicate speedup"""
    if baseline_value == 0:
        return np.nan  # Avoid division by zero
    return ((comparison_value - baseline_value) / baseline_value) * 100

def load_statistics_data(statistics_dir):
    """Load the statistics data for all categories."""
    data = {}
    categories = ['baseline', 'counter', 'profiler']
    
    for category in categories:
        csv_file = os.path.join(statistics_dir, f"{category}.csv")
        if os.path.exists(csv_file):
            print(f"Loading {category} data...")
            data[category] = pd.read_csv(csv_file)
            print(f"  Loaded {len(data[category])} rows")
        else:
            print(f"Warning: {csv_file} not found")
            return None
    
    return data

def compare_implementations(data):
    """Compare counter and profiler against baseline."""
    baseline_df = data['baseline']
    counter_df = data['counter']
    profiler_df = data['profiler']
    
    # Verify all dataframes have the same structure
    if len(baseline_df) != len(counter_df) or len(baseline_df) != len(profiler_df):
        print("Error: Dataframes have different lengths")
        return None
    
    print(f"Comparing {len(baseline_df)} rows across implementations...")
    
    # Start with identifying columns from baseline
    result_df = baseline_df[['run_id', 'host_id', 'pcie', 'core_x', 'core_y', 'risc_type']].copy()
    
    # Add baseline values for reference
    result_df['BASELINE_MEAN'] = baseline_df['KERNEL_LENGTH_AVG']
    result_df['BASELINE_STD'] = baseline_df['KERNEL_LENGTH_STD']
    
    # Add counter values and calculate relative errors
    result_df['COUNTER_MEAN'] = counter_df['KERNEL_LENGTH_AVG']
    result_df['COUNTER_STD'] = counter_df['KERNEL_LENGTH_STD']
    
    # Calculate percentage slowdown for counter vs baseline
    result_df['COUNTER_MEAN_SLOWDOWN_PCT'] = [
        calculate_percentage_slowdown(baseline_df['KERNEL_LENGTH_AVG'].iloc[i], 
                                    counter_df['KERNEL_LENGTH_AVG'].iloc[i])
        for i in range(len(baseline_df))
    ]
    
    result_df['COUNTER_STD_CHANGE_PCT'] = [
        calculate_percentage_slowdown(baseline_df['KERNEL_LENGTH_STD'].iloc[i], 
                                    counter_df['KERNEL_LENGTH_STD'].iloc[i])
        for i in range(len(baseline_df))
    ]
    
    # Add profiler values and calculate relative errors
    result_df['PROFILER_MEAN'] = profiler_df['KERNEL_LENGTH_AVG']
    result_df['PROFILER_STD'] = profiler_df['KERNEL_LENGTH_STD']
    
    # Calculate percentage slowdown for profiler vs baseline
    result_df['PROFILER_MEAN_SLOWDOWN_PCT'] = [
        calculate_percentage_slowdown(baseline_df['KERNEL_LENGTH_AVG'].iloc[i], 
                                    profiler_df['KERNEL_LENGTH_AVG'].iloc[i])
        for i in range(len(baseline_df))
    ]
    
    result_df['PROFILER_STD_CHANGE_PCT'] = [
        calculate_percentage_slowdown(baseline_df['KERNEL_LENGTH_STD'].iloc[i], 
                                    profiler_df['KERNEL_LENGTH_STD'].iloc[i])
        for i in range(len(baseline_df))
    ]
    
    return result_df

def calculate_summary_statistics(comparison_df):
    """Calculate overall summary statistics for the comparison."""
    summary_stats = {}
    
    # Counter vs Baseline
    counter_mean_slowdown = comparison_df['COUNTER_MEAN_SLOWDOWN_PCT'].dropna()
    counter_std_change = comparison_df['COUNTER_STD_CHANGE_PCT'].dropna()
    
    summary_stats['counter'] = {
        'mean_slowdown_avg': counter_mean_slowdown.mean(),
        'mean_slowdown_std': counter_mean_slowdown.std(),
        'mean_slowdown_median': counter_mean_slowdown.median(),
        'std_change_avg': counter_std_change.mean(),
        'std_change_std': counter_std_change.std(),
        'std_change_median': counter_std_change.median(),
        'count': len(counter_mean_slowdown)
    }
    
    # Profiler vs Baseline
    profiler_mean_slowdown = comparison_df['PROFILER_MEAN_SLOWDOWN_PCT'].dropna()
    profiler_std_change = comparison_df['PROFILER_STD_CHANGE_PCT'].dropna()
    
    summary_stats['profiler'] = {
        'mean_slowdown_avg': profiler_mean_slowdown.mean(),
        'mean_slowdown_std': profiler_mean_slowdown.std(),
        'mean_slowdown_median': profiler_mean_slowdown.median(),
        'std_change_avg': profiler_std_change.mean(),
        'std_change_std': profiler_std_change.std(),
        'std_change_median': profiler_std_change.median(),
        'count': len(profiler_mean_slowdown)
    }
    
    return summary_stats

def save_comparison_results(comparison_df, summary_stats, output_dir):
    """Save comparison results to CSV files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save detailed comparison
    detailed_file = os.path.join(output_dir, "implementation_comparison.csv")
    comparison_df.to_csv(detailed_file, index=False)
    print(f"Detailed comparison saved to: {detailed_file}")
    
    # Save summary statistics
    summary_data = []
    
    for impl, stats in summary_stats.items():
        summary_data.extend([
            {
                'implementation': impl.upper(),
                'metric': 'KERNEL_LENGTH_MEAN',
                'statistic': 'avg_slowdown_pct',
                'value': stats['mean_slowdown_avg'],
                'count': stats['count']
            },
            {
                'implementation': impl.upper(),
                'metric': 'KERNEL_LENGTH_MEAN',
                'statistic': 'std_slowdown_pct',
                'value': stats['mean_slowdown_std'],
                'count': stats['count']
            },
            {
                'implementation': impl.upper(),
                'metric': 'KERNEL_LENGTH_MEAN',
                'statistic': 'median_slowdown_pct',
                'value': stats['mean_slowdown_median'],
                'count': stats['count']
            },
            {
                'implementation': impl.upper(),
                'metric': 'KERNEL_LENGTH_STD',
                'statistic': 'avg_change_pct',
                'value': stats['std_change_avg'],
                'count': stats['count']
            },
            {
                'implementation': impl.upper(),
                'metric': 'KERNEL_LENGTH_STD',
                'statistic': 'std_change_pct',
                'value': stats['std_change_std'],
                'count': stats['count']
            },
            {
                'implementation': impl.upper(),
                'metric': 'KERNEL_LENGTH_STD',
                'statistic': 'median_change_pct',
                'value': stats['std_change_median'],
                'count': stats['count']
            }
        ])
    
    summary_df = pd.DataFrame(summary_data)
    summary_file = os.path.join(output_dir, "implementation_comparison_summary.csv")
    summary_df.to_csv(summary_file, index=False)
    print(f"Summary statistics saved to: {summary_file}")
    
    # Print summary to console
    print("\n" + "="*60)
    print("IMPLEMENTATION PERFORMANCE IMPACT SUMMARY")
    print("="*60)
    
    for impl, stats in summary_stats.items():
        print(f"\n{impl.upper()} vs BASELINE:")
        print(f"  KERNEL_LENGTH Performance Impact:")
        print(f"    Average Slowdown: {stats['mean_slowdown_avg']:.3f}%")
        print(f"    Std Dev:          {stats['mean_slowdown_std']:.3f}%")
        print(f"    Median Slowdown:  {stats['mean_slowdown_median']:.3f}%")
        print(f"  KERNEL_LENGTH Variability Change:")
        print(f"    Average Change:   {stats['std_change_avg']:.3f}%")
        print(f"    Std Dev:          {stats['std_change_std']:.3f}%")
        print(f"    Median Change:    {stats['std_change_median']:.3f}%")
        print(f"  Data points: {stats['count']}")
        
        # Add interpretation
        if stats['mean_slowdown_avg'] > 0:
            print(f"  → {impl.upper()} is {stats['mean_slowdown_avg']:.2f}% SLOWER than baseline on average")
        else:
            print(f"  → {impl.upper()} is {abs(stats['mean_slowdown_avg']):.2f}% FASTER than baseline on average")

def main():
    """Main function to perform implementation comparison."""
    # Define paths
    statistics_dir = "/Users/sstanisic/code/counter-data/statistics"
    output_dir = "/Users/sstanisic/code/counter-data/comparison"
    
    print("Starting implementation comparison analysis...")
    print(f"Input directory: {statistics_dir}")
    print(f"Output directory: {output_dir}")
    print("-" * 50)
    
    # Load statistics data
    data = load_statistics_data(statistics_dir)
    if data is None:
        print("Failed to load data. Exiting.")
        return
    
    # Perform comparison
    comparison_df = compare_implementations(data)
    if comparison_df is None:
        print("Failed to perform comparison. Exiting.")
        return
    
    # Calculate summary statistics
    summary_stats = calculate_summary_statistics(comparison_df)
    
    # Save results
    save_comparison_results(comparison_df, summary_stats, output_dir)
    
    print("\nComparison analysis complete!")

if __name__ == "__main__":
    main()
