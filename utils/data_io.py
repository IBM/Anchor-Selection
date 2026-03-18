"""
Data I/O Module

This module handles all file input/output operations and data loading for the
correlation analysis system, including:
- Reading evaluation results from local files or HuggingFace datasets
- Parsing verdict data from JSON files
- Extracting model lists and UIDs from evaluation files
- Loading human preference data from Arena datasets
"""

import json
import os
import re
from typing import Dict, List, Optional, Set

from datasets import load_dataset


def get_models_list(exp_path: str) -> List[str]:
    """
    Extract list of all models from evaluation files in a directory.

    Args:
        exp_path: Path to directory containing evaluation JSON files

    Returns:
        Sorted list of unique model names
    """
    models_list = set()
    for filename in os.listdir(exp_path):
        if filename.endswith(".json") and filename.startswith("eval"):
            with open(os.path.join(exp_path, filename), "r", encoding="utf-8") as f:
                data = json.load(f)
                models_list.add(data.get("model_1_name"))
                models_list.add(data.get("model_2_name"))
        elif filename.endswith(".json") and filename.startswith("merged_eval"):
            with open(os.path.join(exp_path, filename), "r", encoding="utf-8") as f:
                data = json.load(f)
                for item in data:
                    models_list.add(item.get("model_1_name"))
                    models_list.add(item.get("model_2_name"))
    models_list = sorted(list(models_list))
    print("num models:", len(models_list))
    print(models_list)
    return models_list


def get_short_uids(exp_path: str) -> Set[str]:
    """
    Extract unique evaluation UIDs from files in a directory.

    Args:
        exp_path: Path to directory containing evaluation JSON files

    Returns:
        Set of unique short UIDs
    """
    short_uids = set()
    for filename in os.listdir(exp_path):
        if filename.endswith(".json") and filename.startswith("eval"):
            if filename.startswith("eval_newset"):
                match = re.search(r"eval_newset_([a-f0-9]*)_", filename)
            else:
                match = re.search(r"eval_([a-f0-9]{16})_", filename)
            if match:
                short_uids.add(match.group(1))
        elif filename.endswith(".json") and filename.startswith("merged_eval"):
            with open(os.path.join(exp_path, filename), "r", encoding="utf-8") as f:
                data = json.load(f)
                for item in data:
                    if "eval_newset" in item.get("uid"):
                        match = re.search(r"eval_newset_([a-f0-9]*)_", item.get("uid"))
                    else:
                        match = re.search(r"([a-f0-9]{16})_", item.get("uid"))
                    if match:
                        short_uids.add(match.group(1))
    print(short_uids)
    print("num short_uids:", len(short_uids))
    return short_uids


def read_files_into_dict(
    exp_path: Optional[str] = None,
    models_list: Optional[List[str]] = None,
    judge_model: Optional[str] = None,
    dataset_name: Optional[str] = None,
    hf_dataset_id: str = "ibm-research/900K-Judgements",
) -> Dict[str, Dict]:
    """
    Read evaluation data either from local files or HuggingFace dataset.

    Args:
        exp_path: Path to local directory (legacy mode)
        models_list: List of models to filter by
        judge_model: Judge model to filter by (HF mode)
        dataset_name: Dataset name to filter by: 'arena-hard' or 'alpacaeval' (HF mode)
        hf_dataset_id: HuggingFace dataset ID

    Returns:
        Dictionary mapping UIDs to evaluation results
    """
    results_dict = dict()

    # HuggingFace dataset mode
    if judge_model is not None or dataset_name is not None:
        print(f"Loading data from HuggingFace dataset: {hf_dataset_id}")
        print(f"  Judge model: {judge_model}")
        print(f"  Dataset: {dataset_name}")

        # Load dataset from HuggingFace
        dataset = load_dataset(hf_dataset_id, split="train")
        df = dataset.to_pandas()

        # Filter by judge model if specified
        if judge_model:
            df = df[df["judge_model"] == judge_model]
            print(f"  Filtered to {len(df)} rows for judge: {judge_model}")

        # Filter by dataset if specified
        if dataset_name:
            # AlpacaEval: UIDs start with "eval_"
            # Arena-Hard: UIDs don't start with "eval_"
            if dataset_name.lower() == "alpacaeval":
                df = df[df["uid"].str.startswith("eval_")]
                print(f"  Filtered to {len(df)} rows for AlpacaEval dataset")
            elif dataset_name.lower() == "arena-hard":
                df = df[~df["uid"].str.startswith("eval_")]
                print(f"  Filtered to {len(df)} rows for Arena-Hard dataset")

        # Convert to results_dict format
        for _, row in df.iterrows():
            uid = row["uid"]
            if uid.startswith("eval_newset_"):
                uid = uid[len("eval_newset_") :]

            if uid in results_dict:
                continue

            # Extract model names from uid (format: {instance_id}_{model_a}_{model_b})
            parts = uid.split("_")
            if len(parts) >= 3:
                # Reconstruct model names (they may contain underscores)
                model_a = row["model_a_in_prompt"]
                model_b = row["model_b_in_prompt"]

                if models_list is not None and (
                    model_a not in models_list or model_b not in models_list
                ):
                    continue

                results_dict[uid] = {
                    "final_verdict": row["final_verdict"],
                    "model_1_name": model_a,
                    "model_2_name": model_b,
                }

        print(f"Loaded {len(results_dict)} evaluations from HuggingFace")
        return results_dict

    # Legacy local file mode
    if exp_path is None:
        raise ValueError("Either exp_path or (judge_model/dataset_name) must be provided")

    for filename in os.listdir(exp_path):
        if filename.endswith(".json") and filename.startswith("eval"):
            with open(os.path.join(exp_path, filename), "r", encoding="utf-8") as f:
                data = json.load(f)
                uid = data.get("uid")
                if uid.startswith("eval_newset_"):
                    uid = uid[len("eval_newset_") :]
                if uid in results_dict:
                    continue
                if models_list is not None and (
                    (data.get("model_1_name") not in models_list)
                    or (data.get("model_2_name") not in models_list)
                ):
                    continue
                results_dict[uid] = {
                    "final_verdict": data.get("final_verdict"),
                    "model_1_name": data.get("model_1_name"),
                    "model_2_name": data.get("model_2_name"),
                }
        elif filename.endswith(".json") and filename.startswith("merged_eval"):
            with open(os.path.join(exp_path, filename), "r", encoding="utf-8") as f:
                data = json.load(f)
                for item in data:
                    uid = item.get("uid")
                    if uid.startswith("eval_newset_"):
                        uid = uid[len("eval_newset_") :]
                    if uid in results_dict:
                        continue
                    if models_list is not None and (
                        (item.get("model_1_name") not in models_list)
                        or (item.get("model_2_name") not in models_list)
                    ):
                        continue
                    results_dict[uid] = {
                        "final_verdict": item.get("final_verdict"),
                        "model_1_name": item.get("model_1_name"),
                        "model_2_name": item.get("model_2_name"),
                    }
    print("len(results_dict)", len(results_dict))
    return results_dict


def parse_verdict(uid: str, results_dict: Dict, verbose: bool = True) -> Optional[tuple]:
    """
    Parse a verdict from the results dictionary.

    Args:
        uid: Unique identifier for the evaluation
        results_dict: Dictionary containing evaluation results
        verbose: Whether to print error messages

    Returns:
        Tuple of ((winner, loser), margin) or None if error
        where margin is one of: '>>', '>', '='
    """
    if uid not in results_dict:
        if verbose:
            print(f"error! {uid} is missing")
        return None
    data = results_dict[uid]

    final_verdict = data.get("final_verdict", "")
    model_1 = data.get("model_1_name")
    model_2 = data.get("model_2_name")

    if not all([model_1, model_2, final_verdict]):
        if verbose:
            print("error!", model_1, model_2, final_verdict)
        return None

    # Parse the verdict
    if ">>" in final_verdict:
        winner, loser = final_verdict.split(">>")
        margin = ">>"
    elif ">" in final_verdict:
        winner, loser = final_verdict.split(">")
        margin = ">"
    elif "=" in final_verdict:
        winner, loser = final_verdict.split("=")
        margin = "="
    else:
        return None

    winner = winner.strip()
    loser = loser.strip()

    # Create comparison tuple
    if winner == model_1:
        return ((winner, loser), margin)
    elif winner == model_2:
        return ((winner, loser), margin)
    else:
        return None


def read_arena_human_pref_140():
    """
    Load the Arena human preference dataset (140k samples).

    Returns:
        pandas DataFrame with human preference data
    """
    ds = load_dataset("lmarena-ai/arena-human-preference-140k")
    ds = ds["train"].to_pandas()
    models = ds["model_a"].tolist() + ds["model_b"].tolist()
    models = list(set(sorted(models)))
    print("num models in datasets:", len(models))
    print("models in dataset:")
    models.sort()
    for model in models:
        print(model)
    return ds

# Made with Bob
