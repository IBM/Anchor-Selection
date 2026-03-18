"""
Utility script to merge multiple evaluation JSON files into a single file.

This script searches for all JSON files matching the pattern 'eval_*.json' in a
specified directory and combines them into a single JSON array file.

Usage:
    python merge_eval_files.py --input_dir /path/to/eval/files

The merged file will be saved in the same directory with a timestamp.
"""

import json
import glob
import os
import argparse
from datetime import datetime


def merge_eval_files(input_dir: str) -> None:
    """
    Merge all JSON files matching 'eval_*.json' pattern into a single file.
    
    Args:
        input_dir: Directory containing the JSON files to merge
    
    Returns:
        None. Creates a merged JSON file in the input directory.
    
    Output:
        Creates a file named 'merged_eval_<timestamp>.json' containing
        an array of all evaluation objects from individual files.
    
    Example:
        >>> merge_eval_files("/path/to/evaluations")
        Merged 150 files into /path/to/evaluations/merged_eval_20260129_075530.json
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    output_file = os.path.join(input_dir, f"merged_eval_{timestamp}.json")
    all_data = []
    
    # Look for all matching files
    file_pattern = os.path.join(input_dir, "eval_*.json")
    files = glob.glob(file_pattern)
    
    if not files:
        print(f"No matching files found in {input_dir}")
        return
    
    print(f"Found {len(files)} files to merge...")
    
    for file_path in files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                all_data.append(data)
        except json.JSONDecodeError as e:
            print(f"Skipping {file_path}: Invalid JSON - {e}")
        except Exception as e:
            print(f"Skipping {file_path}: {e}")
    
    # Write merged data to output
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)
    
    print(f"Successfully merged {len(all_data)} files into {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge multiple evaluation JSON files into a single file"
    )
    parser.add_argument(
        "--input_dir", 
        type=str, 
        required=True,
        help="Directory containing eval_*.json files to merge"
    )
    args = parser.parse_args()

    merge_eval_files(args.input_dir)
