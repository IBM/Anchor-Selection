"""
Test Correlation Analysis Module

The module supports:
- Kendall's Tau correlation analysis
- Win rate matrix computation
- Bradley-Terry model rankings
- Bootstrap confidence intervals
- Visualization of ranking correlations
- Power analysis simulations
"""

import argparse
import json
import math
import os
import random
import re
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from datasets import load_dataset
from scipy.optimize import curve_fit, minimize
from scipy.stats import kendalltau, norm, pearsonr
from sklearn.linear_model import LogisticRegression

# Import configuration data
from utils.model_config import (
    MMLU,
    arena_pref_140_to_auto,
    artificial_to_auto_names,
    auto_names_to_pretty_names,
    full_matrix_ranking,
    full_matrix_with_humans_corr,
    higher_anchor_based_with_full_matrix_corr,
    higher_anchor_based_with_humans_corr,
    humans_elo_scores,
    humans_to_auto_names,
    liveCodeBench,
    lower_anchor_based_with_full_matrix_corr,
    lower_anchor_based_with_humans_corr,
    mmlu_scores,
    models_not_in_artificial_analysis,
)

# Import Bradley-Terry model
from utils.bradley_terry import BradleyTerryRanker, run_bradley_terry

# Import data I/O functions
from utils.data_io import (
    get_models_list,
    get_short_uids,
    parse_verdict,
    read_arena_human_pref_140,
    read_files_into_dict,
)


# ============================================================================
# Core Functions: Ranking and Correlation Analysis
# ============================================================================


def get_anchor_based_ranking(anchor: str, opponents_scores: Dict[str, float]) -> List[str]:
    """
    Create a ranking based on an anchor model's performance against opponents.

    Higher win rate against the anchor = higher ranking for the opponent.
    The anchor itself gets a middle ranking based on average performance.

    Args:
        anchor: Name of the anchor model
        opponents_scores: Dictionary mapping opponent model names to their win rates against anchor

    Returns:
        List of model names in ranked order (best to worst)
    """
    # Sort opponents by their win rate against the anchor (descending)
    ranked_opponents = sorted(opponents_scores.items(), key=lambda x: x[1], reverse=True)

    # Calculate where to place the anchor based on its average win rate
    # If anchor has low average win rate against others, it should be ranked higher
    avg_opponent_win_rate = sum(opponents_scores.values()) / len(opponents_scores)
    anchor_implied_strength = 1 - avg_opponent_win_rate  # Anchor's implied win rate

    # Create full ranking including anchor
    full_ranking = []
    anchor_placed = False

    for opponent, win_rate in ranked_opponents:
        # If we haven't placed the anchor yet and this opponent's win rate is lower than
        # what would make them better than the anchor, place anchor first
        if not anchor_placed and win_rate < (1 - anchor_implied_strength):
            full_ranking.append(anchor)
            anchor_placed = True
        full_ranking.append(opponent)

    # If anchor wasn't placed yet, it goes at the end (weakest)
    if not anchor_placed:
        full_ranking.append(anchor)

    return full_ranking


def calculate_kendall_tau_two_lists(
    rank_1: List[str], rank_2: List[str]
) -> Tuple[float, float, set]:
    """
    Calculate Kendall's tau correlation between two rankings.

    Args:
        rank_1: First ranking (list of model names in order)
        rank_2: Second ranking (list of model names in order)

    Returns:
        Tuple of (tau correlation, p-value, set of common models)
    """
    rank_1_positions_dict = {model: i + 1 for i, model in enumerate(rank_1)}
    rank_2_positions_dict = {model: i + 1 for i, model in enumerate(rank_2)}
    common_models = set(rank_1_positions_dict.keys()) & set(rank_2_positions_dict.keys())
    rank_1_positions = [rank_1_positions_dict[model] for model in sorted(common_models)]
    rank_2_positions = [rank_2_positions_dict[model] for model in sorted(common_models)]

    # Calculate Kendall's tau
    tau, p_value = kendalltau(rank_1_positions, rank_2_positions)
    return tau, p_value, common_models


def calculate_kendall_tau_correlations(
    overall_ranking: List[str],
    anchor_data: Dict[str, Dict[str, float]],
    use_bt: bool = False,
    results_dict: Optional[Dict] = None,
    short_uids: Optional[set] = None,
) -> List[Dict[str, Any]]:
    """
    Calculate Kendall's tau correlation between overall ranking and each anchor-based ranking.

    Args:
        overall_ranking: List of models in overall ranking order
        anchor_data: Dictionary mapping anchor models to their opponent scores
        use_bt: Whether to use Bradley-Terry model for ranking
        results_dict: Dictionary of evaluation results (required if use_bt=True)
        short_uids: Set of UIDs to use (required if use_bt=True)

    Returns:
        List of dictionaries containing correlation results for each anchor
    """
    results = []

    print("=" * 80)
    print("KENDALL'S TAU CORRELATION ANALYSIS")
    print("=" * 80)
    print(f"Overall Elo Ranking: {' > '.join(overall_ranking)}")
    print("\n")

    for anchor, opponents_scores in anchor_data.items():
        # Get ranking based on this anchor
        if not use_bt:
            anchor_ranking = get_anchor_based_ranking(anchor, opponents_scores)
        else:
            filtered_results_dict = prepare_filtered_results_dict(results_dict, [anchor])
            leaderboard = run_bradley_terry(
                short_uids, models_list, filtered_results_dict, verbose=False, bootstrap_std=False
            )
            anchor_ranking = leaderboard["model"].tolist()
        tau, p_value, common_models = calculate_kendall_tau_two_lists(
            overall_ranking, anchor_ranking
        )

        results.append(
            {
                "Anchor Model": anchor,
                "Kendall Tau": tau,
                "P-value": p_value,
                "Ranking": " > ".join(anchor_ranking),
                "Common Models": len(common_models),
            }
        )

        print(f"Anchor: {anchor}")
        print(f"Ranking: {' > '.join(anchor_ranking)}")
        print(f"Kendall's τ: {tau:.4f} (p-value: {p_value:.4f})")
        print(f"Models compared: {len(common_models)}")
        print("-" * 80)

    return results


def create_summary_table(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Create a summary table of correlations.

    Args:
        results: List of correlation result dictionaries

    Returns:
        DataFrame with sorted correlation results
    """
    df = pd.DataFrame(results)
    df = df.sort_values("Kendall Tau", ascending=False)

    print("\nSUMMARY TABLE - Correlations with Overall Elo Ranking")
    print("=" * 80)

    for _, row in df.iterrows():
        significance = (
            "***"
            if row["P-value"] < 0.001
            else "**"
            if row["P-value"] < 0.01
            else "*"
            if row["P-value"] < 0.05
            else ""
        )
        print(
            f"{row['Anchor Model']:40} | τ = {row['Kendall Tau']:6.3f}{significance:3} | p = {row['P-value']:6.3f}"
        )

    print("\nSignificance levels: * p<0.05, ** p<0.01, *** p<0.001")
    print(f"Average correlation: {df['Kendall Tau'].mean():.3f}")
    print(f"Std deviation: {df['Kendall Tau'].std():.3f}")

    return df


def analyze_ranking_differences(
    overall_ranking: List[str], anchor_data: Dict[str, Dict[str, float]]
) -> None:
    """
    Analyze which models show the most ranking disagreement across different anchors.

    Args:
        overall_ranking: List of models in overall ranking order
        anchor_data: Dictionary mapping anchor models to their opponent scores
    """
    print("\n" + "=" * 80)
    print("RANKING DISAGREEMENT ANALYSIS")
    print("=" * 80)

    model_positions = {model: [] for model in overall_ranking}

    # Collect all positions for each model across different anchor rankings
    for anchor, opponents_scores in anchor_data.items():
        anchor_ranking = get_anchor_based_ranking(anchor, opponents_scores)
        anchor_positions = {model: i + 1 for i, model in enumerate(anchor_ranking)}

        for model in overall_ranking:
            if model in anchor_positions:
                model_positions[model].append(anchor_positions[model])

    # Calculate position variance for each model
    position_variance = {}
    for model, positions in model_positions.items():
        overall_pos = overall_ranking.index(model) + 1
        if positions:  # Only if model appears in anchor rankings
            variance = np.var(positions)
            position_variance[model] = {
                "overall_position": overall_pos,
                "anchor_positions": positions,
                "variance": variance,
                "min_position": min(positions),
                "max_position": max(positions),
                "range": max(positions) - min(positions),
            }

    # Sort by variance (most disagreement first)
    sorted_models = sorted(position_variance.items(), key=lambda x: x[1]["variance"], reverse=True)

    print("Models ranked by position disagreement across anchor rankings:")
    print("-" * 80)

    for model, stats in sorted_models:
        print(
            f"{model:40} | Overall: #{stats['overall_position']:2} | "
            f"Range: #{stats['min_position']:2}-#{stats['max_position']:2} | "
            f"Variance: {stats['variance']:5.2f}"
        )
        positions_str = ", ".join([f"#{p}" for p in sorted(stats["anchor_positions"])])
        print(f"{'':42} Anchor positions: {positions_str}")
        print("-" * 80)


def convert_and_prepare_humans_data(
    elo_scores: Dict[str, int], names_dict: Dict[str, str]
) -> Dict[str, int]:
    """
    Convert human ELO scores to use automated evaluation names.

    Args:
        elo_scores: Dictionary mapping human model names to ELO scores
        names_dict: Dictionary mapping human names to automated names

    Returns:
        Dictionary with automated names and sorted by ELO score (descending)
    """
    converted_data = {}
    for model_name in elo_scores:
        converted_data[names_dict[model_name]] = elo_scores[model_name]
    converted_data_sorted = dict(
        sorted(converted_data.items(), key=lambda item: item[1], reverse=True)
    )
    return converted_data_sorted


def print_human_models() -> None:
    """Print all human model names and their automated equivalents."""
    print("num models:", len(humans_to_auto_names))
    lst = []
    for model_name in humans_to_auto_names:
        auto_name = humans_to_auto_names[model_name]
        print(auto_name, end=" ")
        lst.append(auto_name.lower())
    lst.sort()
    for model_name in lst:
        print(model_name)
    print()


# get_models_list and get_short_uids are now imported from data_io module


def pretty_print_nested_dict(nested_dict: Dict[str, Dict]) -> None:
    """Pretty print a nested dictionary with JSON formatting."""
    for outer_key, inner_dict in nested_dict.items():
        print(f"{outer_key}:")
        print(json.dumps(inner_dict, indent=4))
        print()


def aggregate_win_rates_mean(
    data: Dict[str, Dict[str, float]], anchors: Optional[List[str]] = None
) -> Dict[str, float]:
    models_all = {}
    if anchors:
        anchors_list = anchors
    else:
        anchors_list = data.keys()
    for anchor in anchors_list:
        anchor = anchor.replace("\\/", "/")
        for model in data[anchor]:
            if model in models_all:
                models_all[model].append(data[anchor][model])
            else:
                models_all[model] = [data[anchor][model]]
    # print("models_all", models_all)
    aggregated = {}
    for model in models_all:
        aggregated[model] = np.mean(models_all[model])
    return dict(sorted(aggregated.items(), key=lambda item: item[1], reverse=True))


# read_files_into_dict and parse_verdict are now imported from data_io module


def compare_two(short_uids, model_a, model_b, results_dict):
    model_a_is_better = 0
    model_b_is_better = 0
    tie = 0
    finegrained_score = 0
    for short_uid in short_uids:
        models = sorted([model_a, model_b])
        models = [model.replace("/", "_") for model in models]
        uid = f"{short_uid}_{models[0]}_{models[1]}"
        if uid not in results_dict:
            print(f"error! {uid} is missing from results_dict")
            return None

        result = parse_verdict(uid, results_dict, verbose=False)
        if result is not None:
            (model_1, model_2), margin = result
        else:
            print("error for uid", uid)

        if margin in [">>", ">"]:
            if model_a == model_1:
                model_a_is_better += 1
            elif model_b == model_1:
                model_b_is_better += 1
        elif margin == "=":
            tie += 1
        else:
            print("Watch!", margin)

        if model_b == model_1:
            if margin == ">":
                finegrained_score += 1
            elif margin == ">>":
                finegrained_score += 2
        elif model_a == model_1:
            if margin == ">":
                finegrained_score -= 1
            elif margin == ">>":
                finegrained_score -= 2

    print(f"{model_a} is better than {model_b}: {model_a_is_better/(len(short_uids))}")
    print(f"{model_b} is better than {model_a}: {model_b_is_better/(len(short_uids))}")
    print(f"it's a tie between {model_a} and {model_b}: {tie/(len(short_uids))}\n")
    return (model_b_is_better + (tie / 2)) / len(short_uids), finegrained_score / len(short_uids)


def get_win_rate_for_anchor(short_uids, anchor, models_list, results_dict):
    results = {}
    results_finegrained = {}
    for i in range(len(models_list)):
        model_name = models_list[i]
        if model_name != anchor:
            score, finegrained = compare_two(short_uids, anchor, model_name, results_dict)
            results[model_name] = score
            results_finegrained[model_name] = finegrained
    results_sorted = dict(sorted(results.items(), key=lambda item: item[1]))
    finegrained_results_sorted = dict(sorted(results_finegrained.items(), key=lambda item: item[1]))
    print(f"scores with {anchor} as the anchor model")
    print(json.dumps(results_sorted, indent=4))
    print(json.dumps(finegrained_results_sorted, indent=4))
    print()
    return results_sorted, finegrained_results_sorted


# read_arena_human_pref_140 is now imported from data_io module


def kendall_tau(rank_list_1, rank_list_2):
    """
    Compute Kendall's tau between two ranking lists of names,
    ignoring elements not present in both lists.

    Args:
        rank_list_1 (list): First ranking list (e.g. ["Alice", "Bob", "Charlie"]).
        rank_list_2 (list): Second ranking list (e.g. ["Charlie", "Alice", "Bob"]).

    Returns:
        float: Kendall's tau coefficient, or None if no common elements.
    """
    # Keep only elements that appear in both lists
    common = list(set(rank_list_1) & set(rank_list_2))
    if len(common) < 2:
        return None  # Not enough items to compute Kendall's tau

    # Build rank maps only for common elements
    rank_map_1 = {name: i for i, name in enumerate(rank_list_1) if name in common}
    rank_map_2 = {name: i for i, name in enumerate(rank_list_2) if name in common}

    # Sort by a consistent order of common elements
    elements = sorted(common)
    ranks_1 = [rank_map_1[name] for name in elements]
    ranks_2 = [rank_map_2[name] for name in elements]
    print("ranks_1", ranks_1)
    print("ranks_2", ranks_2)

    tau, _ = kendalltau(ranks_1, ranks_2)
    return tau


def inverse_parabola(x, a, b):
    return a - b * x**2


def fit_inverse_parabola(x, y):
    popt, pcov = curve_fit(inverse_parabola, x, y)
    a, b = popt
    return a, b, pcov


def plot_rank_correlation(
    ranks,
    correlations,
    title="Model Rank vs Correlation",
    xlabel=r"Model rank ($\pi_{quad}$)",
    ylabel=r"Anchor Quality ($\tau_{p, \mathcal{A}}$)",
    figsize=(10, 6),
    show_labels=True,
    u_shape=False,
):
    """
    Create a plot with model rank on X-axis and model correlation on Y-axis.

    Parameters:
    -----------
    ranks : dict
        Dictionary with model names as keys and their ranks as values
    correlations : dict
        Dictionary with model names as keys and their correlations as values
    title : str, optional
        Plot title (default: "Model Rank vs Correlation")
    xlabel : str, optional
        X-axis label (default: "Model Rank")
    ylabel : str, optional
        Y-axis label (default: "Model Correlation")
    figsize : tuple, optional
        Figure size (default: (10, 6))
    show_labels : bool, optional
        Whether to show model names as point labels (default: True)

    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """

    # Find common models in both dictionaries
    common_models = set(ranks.keys()) & set(correlations.keys())

    if not common_models:
        raise ValueError("No common models found in ranks and correlations dictionaries")

    # Extract data for common models
    model_names = list(common_models)
    x_vals = [ranks[model] for model in model_names]
    y_vals = [correlations[model] for model in model_names]

    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)

    # Scatter plot
    color_to_use = "tab:cyan" if "human" in ylabel else "tab:green"
    scatter = ax.scatter(
        x_vals, y_vals, alpha=0.7, s=80, edgecolors="black", linewidth=0.5, color=color_to_use
    )

    # U shape
    # a, b, cov = fit_inverse_parabola(x_vals, y_vals)
    # print(f"a = {a:.4f}, b = {b:.6f}")
    # x_fit = np.linspace(min(x_vals), max(x_vals), 200)
    # y_fit = a - b * x_fit**2 # -(x-a)^2 +b
    X = np.column_stack([np.ones_like(x_vals), x_vals, np.array(x_vals) ** 2])
    coeffs, _, _, _ = np.linalg.lstsq(X, y_vals, rcond=None)
    C0, C1, C2 = coeffs

    # 2. Convert to Vertex Form: y = -c(x - a)^2 + b
    c_scale = -C2
    a_center = C1 / (2 * c_scale)
    b_peak = C0 + c_scale * (a_center**2)

    # --- QUALITY OF FIT MEASURES ---

    # Calculate predicted y values for the known x points
    y_pred = -c_scale * (x_vals - a_center) ** 2 + b_peak

    # Calculate Residuals
    residuals = y_vals - y_pred

    # 1. R-Squared
    ss_res = np.sum(residuals**2)  # Sum of squared residuals
    ss_tot = np.sum((y_vals - np.mean(y_vals)) ** 2)  # Total sum of squares
    r_squared = 1 - (ss_res / ss_tot)

    # 2. RMSE (Root Mean Squared Error)
    rmse = np.sqrt(np.mean(residuals**2))

    print(f"Fit Quality:")
    print(f"R-squared: {r_squared:.4f} (Model explains {r_squared*100:.1f}% of variance)")
    print(f"RMSE:      {rmse:.4f}")

    # --- PLOTTING ---
    x_fit = np.linspace(0, 23, 400)
    y_fit = -c_scale * (x_fit - a_center) ** 2 + b_peak

    # plt.figure(figsize=(8, 6))
    # plt.scatter(x, y, s=40, alpha=0.8, label="Data")
    if u_shape:
        plt.plot(x_fit, y_fit, linewidth=2, color="C1", label="Fitted Model")
        # Add Text Box with Stats
        text_str = "\n".join(
            (
                r"$y = -c(x-a)^2 + b$",
                r"$a=%.2f$" % (a_center,),
                r"$b=%.3f$" % (b_peak,),
                r"$c=%.4f$" % (c_scale,),
                "----------------",
                r"$R^2=%.4f$" % (r_squared,),
                r"$RMSE=%.4f$" % (rmse,),
            )
        )
        props = dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="lightgray")
        plt.text(
            0.05,
            0.35,
            text_str,
            transform=plt.gca().transAxes,
            fontsize=11,
            verticalalignment="top",
            bbox=props,
        )

    # Add visual helper lines for peak
    # plt.vlines(a_center, np.min(y_vals), b_peak, linestyle="--", color="gray", alpha=0.5)
    # plt.hlines(b_peak, 0, a_center, linestyle="--", color="gray", alpha=0.5)

    # Add model labels if requested
    if show_labels:
        for i, model in enumerate(model_names):
            model = auto_names_to_pretty_names.get(model, model)
            offset = (5, 3) if model in {"Qwen3-235B-A22B", "gemma-3-27b-it"} else (5, 5)
            offset = (3, 9) if model == "o1-2024-12-17" else offset
            angle = 45
            # angle = 0
            # if model in {"Gemma 3 27B Instruct", "Llama 4 Maverick Instruct", "o3"}:
            # if model == "Llama 4 Maverick Instruct":
            # offset = (-270, 5)
            ax.annotate(
                model,
                (x_vals[i], y_vals[i]),
                xytext=offset,
                textcoords="offset points",
                fontsize=12,
                alpha=0.8,
                rotation=angle,
            )
        ax.grid(True, alpha=0.3)

    # Customize the plot
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)

    # Add some styling
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    return fig, ax


def get_matrix_humans_correlation(full_matrix_win_rate_dict, human_data_dict):
    common_models = set(full_matrix_win_rate_dict.keys()) & set(human_data_dict.keys())
    overall_common = {
        name: i + 1 for i, name in enumerate(full_matrix_win_rate_dict) if name in common_models
    }
    humans_common = {name: i + 1 for i, name in enumerate(human_data_dict) if name in common_models}
    print(overall_common)
    print(humans_common)
    elements = sorted(common_models)
    ranks_1 = [overall_common[name] for name in elements]
    ranks_2 = [humans_common[name] for name in elements]
    print("Full matrix and humans correlation:", kendalltau(ranks_1, ranks_2))


# BradleyTerryRanker and run_bradley_terry are now imported from bradley_terry module


def aggregate_top(data, ranking):
    top_model = ranking[0]
    aggreagated_win_rate_sorted = aggregate_win_rates_mean(data, anchors=[top_model])
    print("len(aggreagated_win_rate_sorted)", len(aggreagated_win_rate_sorted))
    tau, p_value, common_models = calculate_kendall_tau_two_lists(
        ranking, list(aggreagated_win_rate_sorted.keys())
    )
    return tau


def aggregate_bottom(data, ranking):
    bottom_model = ranking[-1]
    aggreagated_win_rate_sorted = aggregate_win_rates_mean(data, anchors=[bottom_model])
    tau, p_value, common_models = calculate_kendall_tau_two_lists(
        ranking, list(aggreagated_win_rate_sorted.keys())
    )
    return tau


def aggregate_top_and_bottom(data, ranking):
    top_model = ranking[0]
    bottom_model = ranking[-1]
    aggreagated_win_rate_sorted = aggregate_win_rates_mean(data, anchors=[top_model, bottom_model])
    tau, p_value, common_models = calculate_kendall_tau_two_lists(
        ranking, list(aggreagated_win_rate_sorted.keys())
    )
    return tau


def anchor_disctribution(short_uids, results_dict, models, anchors=None):
    if not anchors:
        anchors = models
    for anchor in anchors:
        anchor_stats = {}
        for short_uid in short_uids:
            anchor_stats[short_uid] = 0
            for model in models:
                if model != anchor:
                    sorted_names = sorted([model, anchor])
                    sorted_names = [name.replace("/", "_") for name in sorted_names]
                    key = f"{short_uid}_{sorted_names[0]}_{sorted_names[1]}"
                    result = parse_verdict(key, results_dict)
                    if result is None:
                        print("error!", key, "is missign")
                    (winner, loser), margin = result
                    if margin in [">>", ">"]:
                        if anchor == loser:
                            anchor_stats[short_uid] += 1
        anchor_stats = list(anchor_stats.values())
        plt.figure()
        plt.hist(anchor_stats, edgecolor="black", alpha=0.7, color="tab:purple")
        plt.xlabel("Number of models better than anchor", fontsize=16)
        if anchor == "o3-2025-04-16":
            plt.ylabel("Number of Samples")
        # Remove grid
        plt.grid(False)

        # Remove top and right spines (border lines)
        ax = plt.gca()  # get current axes
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.tight_layout()
        plt.savefig(f'figs/dist_{anchor.replace("/", "_")}.pdf')


def prepare_filtered_results_dict(results_dict, anchors):
    results_dict_for_anchor = {}
    for model in anchors:
        for uid in results_dict:
            if (
                model == results_dict[uid]["model_1_name"]
                or model == results_dict[uid]["model_2_name"]
            ):
                results_dict_for_anchor[uid] = results_dict[uid]
    print("len(results_dict_for_anchors)", len(results_dict_for_anchor), anchors)
    return results_dict_for_anchor


def prepare_filtered_results_dict_per_uids(results_dict, short_uids_set, verbose=True):
    filtered_results_dict = {}
    for uid in results_dict:
        short_uid = uid.split("_")[0]
        if short_uid in short_uids_set:
            filtered_results_dict[uid] = results_dict[uid]
    if verbose:
        print("len(filtered_results_dict)", len(filtered_results_dict))
    return filtered_results_dict


def split_results_dict_evenly(short_uids, results_dict, anchors, seed=None):
    short_uids = list(short_uids)
    num_samples_per_anchor = int(len(short_uids) / len(anchors))
    # print("num_samples_per_anchor", num_samples_per_anchor)
    rng = random.Random(seed)  # Create seeded random number generator
    short_uids_ordered = rng.sample(short_uids, len(short_uids))
    results_dict_filtered = {}
    j = 0
    for i in range(0, len(short_uids_ordered), num_samples_per_anchor):
        short_uids_for_anchor = short_uids_ordered[i : i + num_samples_per_anchor]
        counter_anchor = 0
        uids_tracker = []
        for short_uid in short_uids_for_anchor:
            for uid in results_dict:
                if short_uid in uid and (
                    anchors[j] == results_dict[uid]["model_1_name"]
                    or anchors[j] == results_dict[uid]["model_2_name"]
                ):
                    results_dict_filtered[uid] = results_dict[uid]
                    uids_tracker.append(uid)
                    counter_anchor += 1
        j += 1
        if j + 1 > len(anchors):
            break
    return results_dict_filtered


def plot_num_anchors_correlations(
    data, ranking, title="Humans Ranking with Anchor (Mean) ranking Correaltion"
):
    correlations_all = []
    for i in range(2000):
        correlations = []
        rng = random.Random(5 * i)  # Create seeded random number generator
        models = rng.sample(ranking, len(ranking))
        models = [model.replace("\\/", "/") for model in models]
        anchors = []
        for model in models:
            anchors.append(model)
            aggreagated_win_rate_sorted = aggregate_win_rates_mean(data, anchors=anchors)
            tau, p_value, common_models = calculate_kendall_tau_two_lists(
                ranking, list(aggreagated_win_rate_sorted.keys())
            )
            correlations.append(tau)
        correlations_all.append(correlations)
    x = range(1, len(ranking) + 1)
    mean_corr = np.mean(correlations_all, axis=0)
    std_corr = np.std(correlations_all, axis=0)
    plt.plot(x, mean_corr, label="Mean correlation")
    plt.fill_between(x, mean_corr - std_corr, mean_corr + std_corr, alpha=0.2, label="±1 std")
    plt.title(title)
    plt.xlabel("Number of Anchors")
    plt.ylabel("Correlation with Ranking")
    plt.show()


def plot_num_anchors_correlations_bt(
    short_uids, ranking, data=None, title="Humans Ranking with Anchor ranking Correaltion"
):
    correlations_all = []
    seeds = 40
    for i in range(seeds):
        correlations = []
        rng = random.Random(5 * i)  # Create seeded random number generator
        models = rng.sample(ranking, len(ranking))
        models = [model.replace("\\/", "/") for model in models]
        anchors = []
        for model in models:
            anchors.append(model)
            filtered_results_dict = prepare_filtered_results_dict(results_dict, anchors)
            leadrboard = run_bradley_terry(
                short_uids,
                models_list,
                filtered_results_dict,
                use_prob=False,
                verbose=False,
                bootstrap_std=False,
            )
            bt_ranking = leadrboard["model"].tolist()
            tau, p_value, common_models = calculate_kendall_tau_two_lists(ranking, bt_ranking)
            correlations.append(tau)
        correlations_all.append(correlations)
    x = range(1, len(ranking) + 1)
    mean_corr = np.mean(correlations_all, axis=0)
    std_corr = np.std(correlations_all, axis=0)
    figsize = (10, 6)
    fig, ax = plt.subplots(figsize=figsize)
    plt.plot(x, mean_corr, label="Mean correlation")
    plt.fill_between(x, mean_corr - std_corr, mean_corr + std_corr, alpha=0.2, label="±1 std")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("Number of Anchors", fontsize=18)
    ax.set_ylabel(r"Mean Anchor Quality ($\tau_{p, \mathcal{A}}$)", fontsize=18)

    plt.legend()
    plt.tight_layout()
    plt.savefig(f"figs/num_anchors{'_humans' if 'Humans' in title else ''}_{seeds}.pdf")
    plt.show()
    print("mean_corr", mean_corr)


def bt_exp(short_uids, models_list, results_dict, humans_ranking):
    short_uids = list(short_uids)
    repeats = 100
    all_tau_full = []
    for i in range(repeats):
        if i > 0:
            # we validated it once, the results are very stable no need to rerun
            break
        rng = random.Random(5 * i)
        short_uids_ordered = rng.sample(short_uids, len(short_uids))
        leadrboard = run_bradley_terry(
            short_uids_ordered, models_list, results_dict, verbose=False, bootstrap_std=False
        )
        bt_ranking = leadrboard["model"].tolist()
        print("bt_ranking", bt_ranking)
        tau_full, p_value, common_models = calculate_kendall_tau_two_lists(
            bt_ranking, humans_ranking
        )
        all_tau_full.append(tau_full)
    # use the last ranking as 'gold-matrix' (again, assuming bt is very stable)
    gold_matrix_ranking = bt_ranking

    all_taus_evenly = []
    all_taus_evenly_gold = []
    for i in range(repeats):
        filtered_results_dict = split_results_dict_evenly(short_uids, results_dict, models_list)
        leadrboard = run_bradley_terry(
            short_uids, models_list, filtered_results_dict, verbose=False, bootstrap_std=False
        )
        bt_ranking = leadrboard["model"].tolist()
        tau, p_value, common_models = calculate_kendall_tau_two_lists(bt_ranking, humans_ranking)
        all_taus_evenly.append(tau)
        tau, p_value, common_models = calculate_kendall_tau_two_lists(
            bt_ranking, gold_matrix_ranking
        )
        all_taus_evenly_gold.append(tau)

    all_taus_one = []
    all_taus_one_gold = []
    for i in range(repeats):
        rng = random.Random(5 * i)
        anchor = rng.sample(models_list, 1)
        filtered_results_dict = prepare_filtered_results_dict(results_dict, anchor)
        leadrboard = run_bradley_terry(
            short_uids, models_list, filtered_results_dict, verbose=False, bootstrap_std=False
        )
        bt_ranking = leadrboard["model"].tolist()
        tau, p_value, common_models = calculate_kendall_tau_two_lists(bt_ranking, humans_ranking)
        all_taus_one.append(tau)
        tau, p_value, common_models = calculate_kendall_tau_two_lists(
            bt_ranking, gold_matrix_ranking
        )
        all_taus_one_gold.append(tau)

    print(
        f"tau with humans for bt on full matrix with {repeats} seeds: {np.mean(all_tau_full)}, bottom: {np.min(all_tau_full)}"
    )
    print(
        f"std for tau with humans for bt on full matrix with {repeats} seeds: {np.std(all_tau_full)}"
    )
    print(
        f"tau with humans for bt on evenly splitted with {repeats} seeds: {np.mean(all_taus_evenly)}, bottom: {np.min(all_taus_evenly)}"
    )
    print(
        f"std for tau with humans for bt on evenly splitted with {repeats} seeds: {np.std(all_taus_evenly)}"
    )
    print(
        f"tau with humans for bt on one anchor a time with {repeats} seeds: {np.mean(all_taus_one)}, bottom: {np.min(all_taus_one)}"
    )
    print(
        f"std for tau with humans for bt on one anchor a time with {repeats} seeds: {np.std(all_taus_one)}"
    )
    print()
    print(
        f"tau with gold for bt on evenly splitted with {repeats} seeds: {np.mean(all_taus_evenly_gold)}, bottom: {np.min(all_taus_evenly_gold)}"
    )
    print(
        f"std for tau with gold for bt on evenly splitted with {repeats} seeds: {np.std(all_taus_evenly_gold)}"
    )
    print(
        f"tau with gold for bt on one anchor a time with {repeats} seeds: {np.mean(all_taus_one_gold)}, bottom: {np.min(all_taus_one_gold)}"
    )
    print(
        f"std for tau with gold for bt on one anchor a time with {repeats} seeds: {np.std(all_taus_one_gold)}"
    )


def common_datapoints_for_pair_and_anchor(
    model_a, model_b, anchor, short_uids, results_dict, simplefy=True, verbose=True
):
    all_fail = 0
    all_win = 0
    all_tie = 0
    diff = 0
    for short_uid in short_uids:
        models_a = [model.replace("/", "_") for model in sorted([model_a, anchor])]
        models_b = [model.replace("/", "_") for model in sorted([model_b, anchor])]
        uid_model_a = f"{short_uid}_{models_a[0]}_{models_a[1]}"
        uid_model_b = f"{short_uid}_{models_b[0]}_{models_b[1]}"
        if uid_model_a not in results_dict or uid_model_b not in results_dict:
            print(f"error! {uid_model_a}/{uid_model_b} not in results_dict")
            exit()
        (winner_a, losser_a), margin_a = parse_verdict(uid_model_a, results_dict)
        (winner_b, losser_b), margin_b = parse_verdict(uid_model_b, results_dict)
        if simplefy:
            margin_a = ">" if margin_a == ">>" else margin_a
            margin_b = ">" if margin_b == ">>" else margin_b
        if margin_a == margin_b == "=":
            all_tie += 1
        elif margin_a == margin_b:
            if winner_a == winner_b:
                all_fail += 1
            elif losser_a == losser_b:
                all_win += 1
            else:
                diff += 1
        else:
            diff += 1
    if verbose:
        print(
            f"% all_fails for models: {model_a}, {model_b} with anchor: {anchor} = {all_fail / len(short_uids)}"
        )
        print(
            f"% all_win for models: {model_a}, {model_b} with anchor: {anchor} = {all_win / len(short_uids)}"
        )
        print(
            f"% all_tie for models: {model_a}, {model_b} with anchor: {anchor} = {all_tie / len(short_uids)}"
        )
        print(
            f"% diff for models: {model_a}, {model_b} with anchor: {anchor} = {diff / len(short_uids)}\n"
        )
    return all_fail, all_win, all_tie, diff


def common_datapoints_exp(
    short_uids, results_dict, models_list, anchors=None, gold_corr=None, verbose=True
):
    if anchors is None:
        anchors = models_list
    stats = {}
    for anchor in anchors:
        if verbose:
            print(f"============== {anchor} ==============")
        anchor_non_common = []
        for i in range(len(models_list)):
            for j in range(i + 1, len(models_list)):
                if anchor == models_list[i] or anchor == models_list[j]:
                    continue
                result = common_datapoints_for_pair_and_anchor(
                    models_list[i],
                    models_list[j],
                    anchor,
                    short_uids,
                    results_dict,
                    simplefy=False,
                    verbose=verbose,
                )
                anchor_non_common.append(result[3] / len(short_uids))
        stats[anchor] = (np.mean(anchor_non_common), np.std(anchor_non_common))
    stats = dict(sorted(stats.items(), key=lambda item: item[1]))
    if verbose:
        for key in stats:
            print(f"{key}: {stats[key]}")
    if gold_corr is not None:
        fig, ax = plt.subplots(figsize=(10, 6))
        x_vals = [stats[model_name][0] for model_name in stats]
        y_vals = [gold_corr[model_name] for model_name in stats]
        scatter = ax.scatter(x_vals, y_vals, alpha=0.7, s=80, edgecolors="black", linewidth=0.5)

        # Linear regression
        slope, intercept = np.polyfit(x_vals, y_vals, 1)

        # Regression line
        x_fit = np.linspace(np.min(x_vals), np.max(x_vals), 100)
        y_fit = slope * x_fit + intercept
        print("Linear fit:", slope, x_fit, intercept, y_fit)

        # 1. Calculate Correlation Coefficient (r)
        correlation_matrix = np.corrcoef(x_vals, y_vals)
        r = correlation_matrix[0, 1]

        # 2. Calculate R-squared (R²)
        r_squared = r**2

        print(f"Correlation (r): {r:.4f}")
        print(f"R-squared (R²): {r_squared:.4f}")

        # ax.set_yscale('log')
        # Add model labels if requested
        for i, model in enumerate(stats.keys()):
            pos = (5, 5)
            model = auto_names_to_pretty_names.get(model, model)
            if model == "Qwen2.5 72B Instruct" or model == "Athene V2 Chat":
                pos = (5, -2)
            elif model == "Llama 3.1 Nemotron 70B Instruct":
                pos = (5, 7)
            elif model == "Qwen3 30B A3B" or model == "GPT-4.5 (Preview)":
                pos = (5, 12)
            elif model == "QwQ 32B":
                pos = (-50, 0)
            elif model == "GPT-4.1 Mini":
                pos = (10, 5)
            elif model == "Qwen3 32B":
                pos = (3, 5)
            elif model == "GPT-4.1":
                pos = (-20, 5)
            elif model == "o1" or model == "o3 Mini":
                pos = (5, 2)
            # if model in {"Gemma 3 27B Instruct", "Llama 4 Maverick Instruct", "o3"}:
            # if model == "Llama 4 Maverick Instruct":
            # offset = (-270, 5)
            # ax.annotate(model, (x_vals[i], y_vals[i]),
            #         xytext=pos, textcoords='offset points',
            #         fontsize=20, alpha=0.8, rotation=0)
            # ax.grid(True, alpha=0.3)

            ax.annotate(
                model,
                (x_vals[i], y_vals[i]),
                xytext=pos,
                textcoords="offset points",
                fontsize=9,
                alpha=0.8,
            )
            ax.grid(True, alpha=0.3)

        # Add some styling
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        # ax.set_title('Anchor-based ranking correlation with gold ranking as function of informativity',
        # fontsize=14, pad=20)
        ax.set_xlabel(r"Informativeness ($I(p,\mathcal{A})$)", fontsize=12)
        ax.set_ylabel(r"Anchor Quality ($\tau_{p, \mathcal{A}}$)", fontsize=12)

        plt.tight_layout()
        plt.savefig("figs/corr_vs_informativity_full.pdf")
    return stats


def mean_corr_as_function_of_dataset_size_exp(
    anchor_data, short_uids, results_dict, humans_ranking, use_bt=True
):
    num_examples = [50, 100, 150, 200, 300, 400, 500, 600, 700, 750]
    repeats = 20
    short_uids_set = set(short_uids)
    model_to_corr = {}
    model_to_std = {}
    for model in models_list + ["full_matrix"]:
        model_to_corr[model] = [[0 for i in range(repeats)] for j in range(len(num_examples))]
        model_to_std[model] = [0 for j in range(len(num_examples))]
    for i in range(repeats):
        for j, num in enumerate(num_examples):
            rng = random.Random(5 * i)
            short_uids_to_use = rng.sample(list(short_uids), num)
            filtered_results_dict = prepare_filtered_results_dict_per_uids(
                results_dict, short_uids_set
            )
            leadrboard = run_bradley_terry(
                short_uids_to_use,
                models_list,
                filtered_results_dict,
                verbose=False,
                bootstrap_std=False,
            )
            full_matrix = leadrboard["model"].tolist()
            if humans_ranking is not None:
                overall_ranking = humans_ranking
            else:
                overall_ranking = full_matrix
            tau, p_value, common_models = calculate_kendall_tau_two_lists(
                overall_ranking, full_matrix
            )
            model_to_corr["full_matrix"][j][i] = tau
            for anchor, opponents_scores in anchor_data.items():
                # Get ranking based on this anchor
                if not use_bt:
                    anchor_ranking = get_anchor_based_ranking(anchor, opponents_scores)
                else:
                    filtered_results_dict_for_anchor = prepare_filtered_results_dict(
                        filtered_results_dict, [anchor]
                    )
                    leadrboard = run_bradley_terry(
                        short_uids_to_use,
                        models_list,
                        filtered_results_dict_for_anchor,
                        verbose=False,
                        bootstrap_std=False,
                    )
                    anchor_ranking = leadrboard["model"].tolist()
                tau, p_value, common_models = calculate_kendall_tau_two_lists(
                    overall_ranking, anchor_ranking
                )
                model_to_corr[anchor][j][i] = tau
    fig, ax = plt.subplots(figsize=(10, 6))
    model_to_std = {
        model: [0 for _ in range(len(num_examples))] for model in models_list + ["full_matrix"]
    }
    for model in models_list + ["full_matrix"]:
        for j in range(len(num_examples)):
            model_to_std[model][j] = np.std(model_to_corr[model][j])
            model_to_corr[model][j] = np.mean(model_to_corr[model][j])
        if model == "full_matrix":
            plt.plot(
                num_examples,
                model_to_corr[model],
                label="Quadratic evaluation",
                linestyle=":",
                color="blue",
            )
            plt.fill_between(
                num_examples,
                np.array(model_to_corr[model]) - np.array(model_to_std[model]),
                np.array(model_to_corr[model]) + np.array(model_to_std[model]),
                alpha=0.2,
                label="Quadratic evaluation ±1 std",
                color="blue",
            )
        # elif model == "o3-2025-04-16":
        # else:
        # plt.plot(num_examples, model_to_corr[model], label="o3", color="green")

    average_corr = []
    std_corr = []

    for j in range(len(num_examples)):
        corr_for_j = []
        for model in models_list:
            corr_for_j.append(model_to_corr[model][j])
        average_corr.append(np.mean(corr_for_j))
        std_corr.append(np.std(corr_for_j))
    plt.plot(num_examples, average_corr, label="Mean one anchor", linestyle="--", color="orange")
    plt.fill_between(
        num_examples,
        np.array(average_corr) - np.array(std_corr),
        np.array(average_corr) + np.array(std_corr),
        alpha=0.2,
        label="One anchor ±1 std",
        color="orange",
    )
    plt.plot(num_examples, model_to_corr["o3-2025-04-16"], label="o3", color="green")
    plt.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.xlabel("Number of Samples")
    plt.ylabel(r"Mean Anchor Quality defined by human ($\tau_{p, \mathcal{A}}$)")
    plt.tight_layout()
    plt.savefig(f"figs/num_samples_{filename_suffix}.pdf")


def corr_judges_ranking():
    judges = list(full_matrix_ranking.keys())
    for i in range(len(judges)):
        for j in range(i + 1, len(judges)):
            judge_a = judges[i]
            judge_b = judges[j]
            tau, _, _ = calculate_kendall_tau_two_lists(
                full_matrix_ranking[judge_a], full_matrix_ranking[judge_b]
            )
            print(f"corr {judge_a} with {judge_b} (full matrix):", tau)


def convert_verdict_to_num(verdict, anchor):
    (winner, loser), margin = verdict
    winner = winner.replace("/", "_")
    loser = loser.replace("/", "_")
    if margin == "=":
        model_score = 0
    elif margin == ">":
        if anchor == loser:
            model_score = 1
        elif anchor == winner:
            model_score = -1
        else:
            print("error! margin is not valid", verdict)
    elif margin == ">>":
        if anchor == loser:
            model_score = 2
        elif anchor == winner:
            model_score = -2
        else:
            print("error! margin is not valid", verdict)
    return model_score


def construct_dists_for_anchor(short_uids, results_dict, models_list, anchor):
    all_dists = {}
    for i in range(len(models_list)):
        model_a = models_list[i].replace("/", "_")
        if model_a == anchor:
            continue
        for j in range(i + 1, len(models_list)):
            model_b = models_list[j].replace("/", "_")
            if model_b == anchor:
                continue
            all_dists[f"{model_a}_{model_b}"] = []
            for short_uid in short_uids:
                uid_model_a = (
                    f"{short_uid}_{model_a}_{anchor}"
                    if f"{short_uid}_{model_a}_{anchor}" in results_dict
                    else f"{short_uid}_{anchor}_{model_a}"
                )
                result = parse_verdict(uid_model_a, results_dict)
                if result is None:
                    print("error!", uid_model_a, "is missign")
                    exit()
                model_a_score = convert_verdict_to_num(result, anchor)
                uid_model_b = (
                    f"{short_uid}_{model_b}_{anchor}"
                    if f"{short_uid}_{model_b}_{anchor}" in results_dict
                    else f"{short_uid}_{anchor}_{model_b}"
                )
                result = parse_verdict(uid_model_b, results_dict)
                if result is None:
                    print("error!", uid_model_b, "is missign")
                    exit()
                model_b_score = convert_verdict_to_num(result, anchor)
                all_dists[f"{model_a}_{model_b}"].append(model_a_score - model_b_score)
    return all_dists


def get_D():
    # 1. Define the values
    values = np.array([-2, -1, 0, 1, 2])

    # 2. Define Base Distribution P (skewed to have positive mean)
    # Probabilities corresponding to [-2, -1, 0, 1, 2]
    P_probs = np.array([0.05, 0.10, 0.20, 0.35, 0.30])
    mean_P = np.sum(values * P_probs)
    print(f"Theoretical Mean P: {mean_P}")

    # 3. Define Target Mean for Q (5% higher)
    mean_Q_target = mean_P * 1.05
    print(f"Target Mean Q: {mean_Q_target}")

    # 4. Find Distribution Q using optimization
    # We want Q to be close to P but with the new mean
    def objective(q):
        return np.sum((q - P_probs) ** 2)  # Minimize squared difference

    constraints = (
        {"type": "eq", "fun": lambda q: np.sum(q) - 1},  # Sum to 1
        {"type": "eq", "fun": lambda q: np.sum(q * values) - mean_Q_target},  # Target mean
    )
    bounds = [(0, 1) for _ in range(5)]  # Probabilities must be 0-1

    # Solve
    result = minimize(objective, P_probs, constraints=constraints, bounds=bounds)
    Q_probs = result.x / np.sum(result.x)  # Normalize to ensure sum is exactly 1

    print("\nCalculated Probabilities for Q:")
    print(Q_probs)

    # 5. Simulate Sampling
    n_samples = 100000
    samples_P = np.random.choice(values, size=n_samples, p=P_probs)
    samples_Q = np.random.choice(values, size=n_samples, p=Q_probs)

    # 6. Calculate Empirical Means
    emp_mean_P = np.mean(samples_P)
    emp_mean_Q = np.mean(samples_Q)

    print(f"\nEmpirical Mean P (N={n_samples}): {emp_mean_P:.5f}")
    print(f"Empirical Mean Q (N={n_samples}): {emp_mean_Q:.5f}")
    print(f"Observed % Difference: {(emp_mean_Q - emp_mean_P)/emp_mean_P * 100:.2f}%")
    return P_probs, Q_probs


def run_power_anaylsis_simulation(N, alpha, p, M, D):
    print(N)
    null_hypothesis_rejected = 0
    clean_D = []
    zeros_ratio = []
    for d in D:
        pos_minus_neg = len([1 for x in d if x > 0]) - len([1 for x in d if x < 0])
        if pos_minus_neg < 0:
            d = [-x for x in d]
        pos_ratio = 1.0 * len([1 for x in d if x > 0]) / len([1 for x in d if x != 0])
        if pos_ratio > p[0] and pos_ratio <= p[1]:
            clean_D.append(d)
            zeros_ratio.append(len([1 for x in d if x == 0]) / len(d))
    print("zeros_ratio", np.mean(zeros_ratio))
    print("len(clean_D)", len(clean_D))

    for m in range(M):
        i = np.random.choice(len(clean_D), replace=False)
        D_i = clean_D[i]
        S = np.random.choice(D_i, N, replace=True)
        res = stats.wilcoxon(S, alternative="greater", zero_method="pratt")
        if res.pvalue < alpha:
            null_hypothesis_rejected += 1
    return null_hypothesis_rejected / M


def compare_two_benchmarks(artificial_dict, benchmark_1, benchmark_2):
    benchmark_1_results = []
    benchmark_2_results = []
    common_models = []
    for model in artificial_dict:
        if (
            benchmark_1 in artificial_dict[model]["evaluations"]
            and benchmark_2 in artificial_dict[model]["evaluations"]
        ):
            if (
                artificial_dict[model]["evaluations"][benchmark_1] is not None
                and artificial_dict[model]["evaluations"][benchmark_2] is not None
            ):
                benchmark_1_results.append(artificial_dict[model]["evaluations"][benchmark_1])
                benchmark_2_results.append(artificial_dict[model]["evaluations"][benchmark_2])
                common_models.append(model)
    print(f"corr {benchmark_1} with {benchmark_2}")
    print("pearson", pearsonr(benchmark_1_results, benchmark_2_results).statistic)
    print("kendalltau", kendalltau(benchmark_1_results, benchmark_2_results).statistic)
    print()


def read_artificial_analysis_results(bt_scores=None):
    file_path = "/Users/shachardon/models.json"
    with open(file_path, "r") as file:
        data_list = json.load(file)["data"]
    models_dict = {item["name"]: item for item in data_list}
    models_dict = {
        atrificial_to_auto_names[model_name]: models_dict[model_name]
        for model_name in atrificial_to_auto_names
    }
    if bt_scores is not None:
        for model_name in models_dict:
            models_dict[model_name]["evaluations"]["bt_score"] = bt_scores[
                model_name.replace("\\/", "/")
            ]
        for model_name in humans_elo_scores:
            if humans_to_auto_names[model_name] in models_dict:
                models_dict[humans_to_auto_names[model_name]]["evaluations"][
                    "arena_score"
                ] = humans_elo_scores[model_name]

    # benchmarks = ["gpqa", "hle", "ifbench", "mmlu_pro", "livecodebench"]
    benchmarks = ["gpqa", "ifbench", "mmlu_pro"]
    all_benchmarks_ranking = {}
    for benchmark in benchmarks:
        benchmark_result = {}
        for model in models_dict:
            if (
                benchmark in models_dict[model]["evaluations"]
                and models_dict[model]["evaluations"][benchmark] is not None
            ):
                benchmark_result[model] = models_dict[model]["evaluations"][benchmark]
        print(benchmark, benchmark_result)
        benchmark_sorted = sorted(benchmark_result.items(), key=lambda item: item[1])
        benchmark_ranking = {}
        for i in range(len(benchmark_sorted)):
            benchmark_ranking[benchmark_sorted[i][0]] = i
        all_benchmarks_ranking[benchmark] = benchmark_ranking
    models_mean_rank = {}
    for model in models_dict:
        models_mean_rank[model] = []
        for benchmark in all_benchmarks_ranking:
            print("all_benchmarks_ranking[benchmark]", all_benchmarks_ranking[benchmark])
            if model in all_benchmarks_ranking[benchmark]:
                models_mean_rank[model].append(all_benchmarks_ranking[benchmark][model])
        models_mean_rank[model] = (
            np.mean(models_mean_rank[model]),
            np.std(models_mean_rank[model]),
        )
    models_mean_rank = sorted(models_mean_rank.items(), key=lambda item: item[1][0])
    for item in models_mean_rank:
        print(item)

    benchmarks = set()
    for model in models_dict:
        for becnmark in models_dict[model]["evaluations"]:
            benchmarks.add(becnmark)
    benchmarks = sorted(list(benchmarks))
    benchmarks = ["arena_score", "bt_score", "gpqa", "hle", "ifbench", "mmlu_pro"]
    print(benchmarks)
    # exit()
    for i in range(len(benchmarks)):
        for j in range(i + 1, len(benchmarks)):
            compare_two_benchmarks(models_dict, benchmarks[i], benchmarks[j])


def find_good_anchor(results_dict, short_uids, num_samples, models_list, anchors=None):
    if anchors is None:
        anchors = models_list
    stats = {anchor: [] for anchor in anchors}
    for i in range(1):
        # rng = random.Random(5 * i)
        short_uids_to_use = random.sample(list(short_uids), num_samples)
        filtered_results_dict = prepare_filtered_results_dict_per_uids(
            results_dict, short_uids_to_use, verbose=False
        )

        for anchor in anchors:
            anchor_non_common = []
            for i in range(len(models_list)):
                for j in range(i + 1, len(models_list)):
                    if anchor == models_list[i] or anchor == models_list[j]:
                        continue
                    result = common_datapoints_for_pair_and_anchor(
                        models_list[i],
                        models_list[j],
                        anchor,
                        short_uids_to_use,
                        filtered_results_dict,
                        simplefy=False,
                        verbose=False,
                    )
                    anchor_non_common.append(result[3] / num_samples)
            stats[anchor].append(np.mean(anchor_non_common))

    stats = {anchor: np.mean(stats[anchor]) for anchor in anchors}
    stats = dict(sorted(stats.items(), key=lambda item: item[1]))
    return stats


def average_dicts(dicts_list):
    # Collect all values for each model across all dictionaries
    model_values = {}
    for d in dicts_list:
        for model, value in d.items():
            if model not in model_values:
                model_values[model] = []
            model_values[model].append(value)

    # Calculate mean and standard deviation for each model
    results = {}
    for model, values in model_values.items():
        mean = np.mean(values)
        std = np.std(values) if len(values) > 1 else 0
        results[model] = {"mean": mean, "std": std, "count": len(values)}
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--show_human_models_names", action="store_true")
    parser.add_argument("--exp_dir", type=str, default=None, help="Legacy: Path to local experiment directory")
    parser.add_argument("--judge_model", type=str, default=None, help="Judge model to use (e.g., 'deepseek-ai/DeepSeek-V3')")
    parser.add_argument("--dataset", type=str, default=None, choices=["arena-hard", "alpacaeval"], help="Dataset to use")
    parser.add_argument("--models_list", nargs="+", default=None)
    parser.add_argument("--show_plots", action="store_true")
    parser.add_argument("--bt_exp", action="store_true")
    parser.add_argument("--num_anchors_exp", action="store_true")
    parser.add_argument("--anchor_dist_exp", action="store_true")
    parser.add_argument("--common_datapoints", action="store_true")
    parser.add_argument("--num_samples", action="store_true")
    parser.add_argument("--judges", action="store_true")
    parser.add_argument("--power_analysis", action="store_true")
    parser.add_argument("--artificail_anaylsis", action="store_true")
    parser.add_argument("--u_shape", action="store_true")

    args = parser.parse_args()

    # Create figs directory if it doesn't exist
    os.makedirs('figs', exist_ok=True)

    if args.show_human_models_names:
        print_human_models()
        exit()

    if args.judges:
        corr_judges_ranking()
        exit()

    # Determine if using HuggingFace or local mode
    use_hf = args.judge_model is not None or args.dataset is not None

    if use_hf and args.exp_dir:
        print(
            "Warning: Both HuggingFace parameters and --exp_dir specified. Using HuggingFace mode."
        )

    if not use_hf and not args.exp_dir:
        parser.error("Either --exp_dir or (--judge_model and/or --dataset) must be specified")

    # Create filename suffix for plots
    if use_hf:
        # Generate suffix from judge model and dataset
        judge_suffix = args.judge_model.split("/")[-1] if args.judge_model else "all_judges"
        dataset_suffix = args.dataset if args.dataset else "all_datasets"
        filename_suffix = f"{judge_suffix}_{dataset_suffix}"
    else:
        # Use last 3 chars of exp_dir as before
        filename_suffix = args.exp_dir[-3:]

    # Get models list
    if args.models_list:
        models_list = args.models_list
    elif use_hf:
        models_list = None  # Will be populated after loading data
    else:
        models_list = get_models_list(args.exp_dir)

    if models_list:
        models_list = [model.replace("\\/", "/") for model in models_list]
        models_list.sort()
        print("len(models_list)", len(models_list))

    # Read files into dict
    if use_hf:
        results_dict = read_files_into_dict(
            exp_path=None,
            models_list=models_list,
            judge_model=args.judge_model,
            dataset_name=args.dataset,
        )

        # Extract models list from loaded data if not provided
        if models_list is None:
            models_set = set()
            for uid, data in results_dict.items():
                models_set.add(data["model_1_name"])
                models_set.add(data["model_2_name"])
            models_list = sorted(list(models_set))
            print(f"Extracted {len(models_list)} models from data")
    else:
        results_dict = read_files_into_dict(args.exp_dir, models_list)

    print(f"Loaded {len(results_dict)} evaluations")

    # Collect uids (short_uid x model_1 x model_2)
    uids = []
    if use_hf:
        # Extract short_uids from the loaded data
        short_uids_set = set()
        for uid in results_dict.keys():
            # Extract short_uid (first part before model names)
            parts = uid.split("_")
            if len(parts) > 0:
                short_uids_set.add(parts[0])
        short_uids = sorted(list(short_uids_set))
        print(f"Extracted {len(short_uids)} unique short UIDs")
    else:
        short_uids = get_short_uids(args.exp_dir)
    for short_uid in short_uids:
        for i in range(len(models_list)):
            for j in range(i + 1, len(models_list)):
                uid = f'{short_uid}_{models_list[i].replace("/", "_")}_{models_list[j].replace("/", "_")}'
                if uid not in results_dict:
                    uid_reorder = f'{short_uid}_{models_list[j].replace("/", "_")}_{models_list[i].replace("/", "_")}'
                    if uid_reorder in results_dict:
                        results_dict[uid] = results_dict[uid_reorder]
                        del results_dict[uid_reorder]
                    else:
                        print("error!", uid, "not in results_dict")
                        # exit()
                uids.append(uid)

    if args.power_analysis:
        all_N = list(range(700, 1100, 10))
        repeat = 10000
        alpha = 0.05
        # effect_size = (0.05, 0.1)
        effects = [(0.55, 0.56), (0.60, 0.61), (0.65, 0.66), (0.70, 0.71), (0.75, 0.76)]
        for effect_size in effects:
            print("effect_size", effect_size)
            D = []
            for anchor in models_list:
                anchor = anchor.replace("/", "_")
                D_for_anchor = construct_dists_for_anchor(
                    short_uids, results_dict, models_list, anchor
                )
                for key in D_for_anchor:
                    D.append(D_for_anchor[key])
            # print(D)

            for N in all_N:
                res = run_power_anaylsis_simulation(N, alpha, effect_size, repeat, D)
                print("1-beta =", res)
                if res >= 0.8:
                    print("we succsefully rejected the null hypothesis with N =", N)
            print("====================")
        exit(0)

    if args.anchor_dist_exp:
        anchor_disctribution(short_uids, results_dict, models_list)
        exit()

    # get the win-rate data
    data = {}
    for model in models_list:
        results_sorted, finegrained_results_sorted = get_win_rate_for_anchor(
            short_uids, model, models_list, results_dict
        )
        data[model] = results_sorted
        # data[model] = finegrained_results_sorted
    pretty_print_nested_dict(data)
    aggreagated_win_rate_sorted = aggregate_win_rates_mean(data)
    print(aggreagated_win_rate_sorted)

    leadrboard = run_bradley_terry(
        short_uids, models_list, results_dict, verbose=False, bootstrap_std=False
    )
    print(leadrboard)
    overall_ranking = leadrboard["model"].tolist()

    if args.artificail_anaylsis:
        # model_to_bt_score = dict(zip(leadrboard['model'].tolist(), leadrboard['rating'].tolist()))
        # read_artificial_analysis_results(model_to_bt_score)
        # exit()

        # num_samples = list(range(10, 50, 10)) + [750]
        num_samples_small = 10
        for i in range(2, 23):
            # print(f"ranking {i} models")
            dicts_small = []
            dicts_all = []
            for j in range(30):
                rng = random.Random(5 * j)
                models_to_rank = rng.sample(models_list, i)
                stats = find_good_anchor(
                    results_dict, short_uids, num_samples_small, models_to_rank, anchors=models_list
                )
                dicts_small.append(stats)

                stats = common_datapoints_exp(
                    short_uids,
                    results_dict,
                    models_to_rank,
                    anchors=models_list,
                    gold_corr=None,
                    verbose=False,
                )
                dicts_all.append(stats)
            small_results = average_dicts(dicts_small)
            # print(small_results)
            all_results = average_dicts(dicts_all)
            pearson, p_value = pearsonr(
                [small_results[anchor]["mean"] for anchor in small_results],
                [all_results[anchor]["mean"] for anchor in small_results],
            )
            print(f"informativeness pearson corr when ranking {i} models: {pearson}")
            small_results = sorted(small_results.items(), key=lambda x: x[1]["mean"], reverse=True)
            # print(small_results)
            for item in small_results:
                print(f"{item[0]}: {item[1]['mean']}, {item[1]['std']}")
            print("=============")
        exit()

    converted_human_data_sorted = convert_and_prepare_humans_data(
        humans_elo_scores, humans_to_auto_names
    )
    humans_ranking = list(converted_human_data_sorted.keys())

    if args.num_anchors_exp:
        print("top correlation with humans", aggregate_top(data, humans_ranking))
        print("bottom correlation with humans", aggregate_bottom(data, humans_ranking))
        print(
            "top and bottom correlation with humans", aggregate_top_and_bottom(data, humans_ranking)
        )
        print()
        print("top correlation with gold", aggregate_top(data, overall_ranking))
        print("bottom correlation with gold", aggregate_bottom(data, overall_ranking))
        print(
            "top and bottom correlation with gold", aggregate_top_and_bottom(data, overall_ranking)
        )
        # plot_num_anchors_correlations(data, humans_ranking)
        # plot_num_anchors_correlations(data, overall_ranking, title="Full Matrix Ranking with Anchor (Mean) ranking Correaltion")
        # plot_num_anchors_correlations_bt(short_uids, humans_ranking, data)
        plot_num_anchors_correlations_bt(
            short_uids,
            overall_ranking,
            data,
            title="Full Matrix Ranking with Anchor ranking Correaltion",
        )  #
        exit()

    # Run the analysis against full matrix ranking
    results = calculate_kendall_tau_correlations(
        overall_ranking, data, use_bt=False, results_dict=results_dict, short_uids=short_uids
    )
    summary_df = create_summary_table(results)
    analyze_ranking_differences(overall_ranking, data)
    correlations = {}
    for item in results:
        correlations[item["Anchor Model"]] = item["Kendall Tau"]
    if args.show_plots:
        pos = {model: i + 1 for (i, model) in enumerate(overall_ranking)}
        fig, ax = plot_rank_correlation(pos, correlations, u_shape=args.u_shape)
        plt.savefig(f"figs/rank_corr_{filename_suffix}{'_u' if args.u_shape else ''}.pdf")
        plt.figure()
        # plt.show()

    # Run the anaysis against humans ranking
    results = calculate_kendall_tau_correlations(
        humans_ranking, data, use_bt=False, results_dict=results_dict, short_uids=short_uids
    )
    summary_df = create_summary_table(results)
    analyze_ranking_differences(humans_ranking, data)
    humans_correlations = {}
    for item in results:
        humans_correlations[item["Anchor Model"]] = item["Kendall Tau"]
    print(humans_correlations)
    if args.show_plots:
        humans_pos = {model: i + 1 for (i, model) in enumerate(humans_ranking)}
        fig, ax = plot_rank_correlation(
            humans_pos,
            humans_correlations,
            xlabel=r"Model rank ($\pi_{human}$)",
            ylabel=r"Anchor Quality defined by human ($\tau_{p, \mathcal{A}}$)",
            u_shape=args.u_shape,
        )
        plt.savefig(f"figs/rank_corr_humans_{filename_suffix}{'_u' if args.u_shape else ''}.pdf")

    # get correlation of humans with full-matrix
    # get_matrix_humans_correlation(aggreagated_win_rate_sorted, converted_human_data_sorted)

    if args.bt_exp:
        bt_exp(short_uids, models_list, results_dict, humans_ranking)

    if args.common_datapoints:
        common_datapoints_exp(
            short_uids, results_dict, models_list, anchors=None, gold_corr=correlations
        )

    if args.num_samples:
        mean_corr_as_function_of_dataset_size_exp(
            data, short_uids, results_dict, humans_ranking, use_bt=True
        )

    print("full matrix ranking =", overall_ranking)
    print(
        "full matrix ranking vs. humans ranking:",
        calculate_kendall_tau_two_lists(overall_ranking, humans_ranking),
    )
    print("higher anchor-based with full matrix corr:", max(list(correlations.values())))
    print("lower anchor-based with full matrix corr:", min(list(correlations.values())))
    print("higher anchor-based with humans corr:", max(list(humans_correlations.values())))
    print("lower anchor-based with humans corr:", min(list(humans_correlations.values())))
