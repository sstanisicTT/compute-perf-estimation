#!/usr/bin/env python3

import pandas as pd
import numpy as np
import os
import argparse

def calculate_std_percentage(input_file, output_file):
    """Calculate standard deviation as percentage of kernel length"""
    print(f"Processing {input_file}...")
    
    # Load the data
    df = pd.read_csv(input_file)
    
    # Check if required columns exist
    required_cols = ['KERNEL_LENGTH_AVG', 'KERNEL_LENGTH_STD']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Calculate standard deviation as percentage of mean
    # Handle division by zero by setting percentage to NaN where mean is 0
    df['KERNEL_LENGTH_STD_PCT'] = np.where(
        df['KERNEL_LENGTH_AVG'] != 0,
        (df['KERNEL_LENGTH_STD'] / df['KERNEL_LENGTH_AVG']) * 100,
        np.nan
    )
    
    # Round to reasonable precision
    df['KERNEL_LENGTH_STD_PCT'] = df['KERNEL_LENGTH_STD_PCT'].round(4)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save the enhanced data
    df.to_csv(output_file, index=False)
    
    # Print some statistics
    valid_pct = df['KERNEL_LENGTH_STD_PCT'].dropna()
    zero_mean_count = (df['KERNEL_LENGTH_AVG'] == 0).sum()
    
    print(f"  Processed {len(df)} rows")
    print(f"  Zero mean values: {zero_mean_count}")
    print(f"  Valid percentage calculations: {len(valid_pct)}")
    
    if len(valid_pct) > 0:
        print(f"  STD Percentage statistics:")
        print(f"    Mean: {valid_pct.mean():.4f}%")
        print(f"    Median: {valid_pct.median():.4f}%")
        print(f"    Min: {valid_pct.min():.4f}%")
        print(f"    Max: {valid_pct.max():.4f}%")
        print(f"    25th percentile: {valid_pct.quantile(0.25):.4f}%")
        print(f"    75th percentile: {valid_pct.quantile(0.75):.4f}%")
    
    return df

def process_all_statistics(input_dir, output_dir):
    """Process all statistics files and calculate percentages"""
    categories = ['baseline', 'counter', 'profiler']
    
    print(f"Processing statistics from {input_dir} to {output_dir}")
    print("=" * 60)
    
    results = {}
    
    for category in categories:
        input_file = os.path.join(input_dir, f"{category}.csv")
        output_file = os.path.join(output_dir, f"{category}.csv")
        
        if not os.path.exists(input_file):
            print(f"Warning: {input_file} not found, skipping...")
            continue
        
        try:
            df = calculate_std_percentage(input_file, output_file)
            results[category] = df
            print(f"  Saved to: {output_file}")
            
        except Exception as e:
            print(f"  Error processing {category}: {e}")
        
        print()
    
    return results

def create_summary_report(results, output_dir):
    """Create a summary report comparing std percentages across categories"""
    if not results:
        print("No results to summarize")
        return
    
    summary_file = os.path.join(output_dir, "std_percentage_summary.txt")
    
    with open(summary_file, 'w') as f:
        f.write("Standard Deviation Percentage Summary Report\n")
        f.write("=" * 50 + "\n\n")
        
        for category, df in results.items():
            valid_pct = df['KERNEL_LENGTH_STD_PCT'].dropna()
            zero_mean_count = (df['KERNEL_LENGTH_AVG'] == 0).sum()
            
            f.write(f"{category.upper()} STATISTICS:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total rows: {len(df)}\n")
            f.write(f"Zero mean values: {zero_mean_count}\n")
            f.write(f"Valid percentage calculations: {len(valid_pct)}\n")
            
            if len(valid_pct) > 0:
                f.write(f"STD Percentage statistics:\n")
                f.write(f"  Mean: {valid_pct.mean():.4f}%\n")
                f.write(f"  Median: {valid_pct.median():.4f}%\n")
                f.write(f"  Standard Deviation: {valid_pct.std():.4f}%\n")
                f.write(f"  Min: {valid_pct.min():.4f}%\n")
                f.write(f"  Max: {valid_pct.max():.4f}%\n")
                f.write(f"  25th percentile: {valid_pct.quantile(0.25):.4f}%\n")
                f.write(f"  75th percentile: {valid_pct.quantile(0.75):.4f}%\n")
                f.write(f"  95th percentile: {valid_pct.quantile(0.95):.4f}%\n")
                f.write(f"  99th percentile: {valid_pct.quantile(0.99):.4f}%\n")
            
            f.write("\n")
        
        # Cross-category comparison
        if len(results) > 1:
            f.write("CROSS-CATEGORY COMPARISON:\n")
            f.write("-" * 25 + "\n")
            
            all_stats = {}
            for category, df in results.items():
                valid_pct = df['KERNEL_LENGTH_STD_PCT'].dropna()
                if len(valid_pct) > 0:
                    all_stats[category] = {
                        'mean': valid_pct.mean(),
                        'median': valid_pct.median(),
                        'std': valid_pct.std(),
                        'count': len(valid_pct)
                    }
            
            if all_stats:
                f.write("Mean STD Percentage by category:\n")
                for category, stats in all_stats.items():
                    f.write(f"  {category}: {stats['mean']:.4f}% (n={stats['count']})\n")
                
                f.write("\nMedian STD Percentage by category:\n")
                for category, stats in all_stats.items():
                    f.write(f"  {category}: {stats['median']:.4f}%\n")
    
    print(f"Summary report saved to: {summary_file}")

def main():
    parser = argparse.ArgumentParser(description='Calculate standard deviations as percentage of kernel length')
    parser.add_argument('--input-dir', '-i', default='statistics', 
                      help='Input directory containing statistics CSV files (default: statistics)')
    parser.add_argument('--output-dir', '-o', default='statistics2', 
                      help='Output directory for enhanced files (default: statistics2)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory {args.input_dir} does not exist")
        return 1
    
    try:
        # Process all statistics files
        results = process_all_statistics(args.input_dir, args.output_dir)
        
        # Create summary report
        create_summary_report(results, args.output_dir)
        
        print("=" * 60)
        print(f"Processing complete! Enhanced files saved in: {args.output_dir}")
        print(f"Each file now includes KERNEL_LENGTH_STD_PCT column")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())



