#!/usr/bin/env python3

import os
import glob
import pandas as pd
from pathlib import Path
import argparse
import tempfile

def find_profile_files(base_dir):
    """Find all profile_log_device.csv files in the directory tree, sorted alphabetically, ignoring .logs files"""
    files = []
    for root, _, _ in os.walk(base_dir):
        found_files = glob.glob(os.path.join(root, "profile_log_device.csv"))
        # Filter out .logs files
        filtered_files = [f for f in found_files if '.logs' not in f]
        files.extend(filtered_files)
    return sorted(files)

def extract_run_info(file_index):
    """Generate run identifier based on file order"""
    return f"{file_index + 1}"

def extract_host_id_from_csv(file_path):
    """Extract host_id from CSV file content"""
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Check if this is a .logs file format (has header in line 2)
    if len(lines) >= 3 and 'run host ID' in lines[1]:
        # Parse header to find host_id column index
        header = lines[1].strip().split(',')
        host_id_idx = None
        for i, col in enumerate(header):
            if 'run host ID' in col:
                host_id_idx = i
                break
        
        # Get host_id from first data row
        data_row = lines[2].strip().split(',')
        return data_row[host_id_idx]
    
    # For processed files, read as pandas and get host_id
    df = pd.read_csv(file_path, nrows=1)
    return str(df['host_id'].iloc[0])

def join_files(input_dir, output_dir, category):
    """Join all profile_log_device.csv files from input_dir and save as [category].csv in output_dir"""
    
    # Find all profile_log_device.csv files in the input directory
    files = find_profile_files(input_dir)
    
    if not files:
        print(f"Warning: No profile_log_device.csv files found in {input_dir}")
        return False
    
    print(f"Processing {len(files)} profile_log_device.csv files for {category} category...")
    
    # Store all dataframes
    all_dfs = []
    
    for file_index, file_path in enumerate(files):
        try:
            # Read regular processed files
            df = pd.read_csv(file_path)
            
            # Add run identifier
            run_id = extract_run_info(file_index)
            df['run_id'] = run_id
            
            # Add host_id if not already present
            if 'host_id' not in df.columns:
                host_id = extract_host_id_from_csv(file_path)
                df['host_id'] = host_id
            
            all_dfs.append(df)
            host_id_info = df['host_id'].iloc[0]
            print(f"  Added {len(df)} rows from {run_id} (host_id: {host_id_info})")
            
        except Exception as e:
            print(f"  Error reading {file_path}: {e}")
    
    if not all_dfs:
        print(f"No valid files found in {input_dir}")
        return False
    
    # Concatenate all dataframes
    # This will automatically handle different column structures by filling missing columns with NaN
    combined_df = pd.concat(all_dfs, ignore_index=True, sort=False)
    
    # Reorder columns to put identifiers first
    cols = combined_df.columns.tolist()
    identifier_cols = ['run_id', 'host_id', 'pcie', 'core_x', 'core_y', 'risc_type']
    other_cols = [col for col in cols if col not in identifier_cols]
    
    # Ensure all identifier columns exist
    final_cols = []
    for col in identifier_cols:
        if col in cols:
            final_cols.append(col)
    final_cols.extend(other_cols)
    
    combined_df = combined_df[final_cols]
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the combined file
    output_file = os.path.join(output_dir, f"{category}.csv")
    combined_df.to_csv(output_file, index=False)
    
    print(f"Created {output_file} with {len(combined_df)} total rows")
    print(f"  Columns: {', '.join(combined_df.columns)}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Join processed profile files by category')
    parser.add_argument('--input', '-i', default='processed', 
                      help='Input directory containing processed files (default: processed)')
    parser.add_argument('--output', '-o', default='unified', 
                      help='Output directory for unified files (default: unified)')
    
    args = parser.parse_args()
    
    input_root = Path(args.input)
    output_root = Path(args.output)
    
    if not input_root.exists():
        print(f"Error: Input directory {input_root} does not exist")
        return

    categories = ['baseline', 'counter', 'profiler']
    success_count = 0
    
    for category in categories:
        print(f"\n{'='*50}")
        print(f"Processing {category.upper()} category")
        print(f"{'='*50}")
        
        category_dir = input_root / category
        if not category_dir.exists():
            continue

        for run_dir in category_dir.iterdir():
            if not run_dir.is_dir():
                continue

            run = run_dir.name

            input_dir = run_dir

            output_dir = output_root / category / run
            os.makedirs(output_dir, exist_ok=True)

            success = join_files(input_dir, output_dir, category)
            
            if success:
                success_count += 1


    print(f"\n{'='*50}")
    print(f"SUMMARY: Successfully processed {success_count}/{len(categories)} categories")
    print(f"Output files created in: {output_dir}")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
