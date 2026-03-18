"""
Model Configuration Data

This module contains all hardcoded configuration data for model evaluation,
including:
- Model name mappings between different systems
- ELO scores and benchmark results
- Pre-computed rankings and correlation statistics
"""

from typing import Dict, List, Optional

# ============================================================================
# Model Name Mappings
# ============================================================================

# Human ELO scores from arena evaluations
# Maps model names to their human-evaluated ELO ratings
humans_elo_scores: Dict[str, int] = {
    "o3-2025-04-16": 1446,
    "o4-mini-2025-04-16": 1393,
    "gemini-2.5-flash": 1407,
    "o3-mini": 1347,
    "o3-mini-high": 1363,
    "o1-2024-12-17": 1399,
    "Qwen3-235B-A22B": 1373,
    "gpt-4.5-preview-2025-02-27": 1439,
    "gpt-4.1-2025-04-14": 1409,
    "gpt-4.1-mini-2025-04-14": 1377,
    "Qwen3-32B": 1346,
    "QwQ-32B": 1335,
    "Qwen3-30B-A3B": 1328,
    "athene-v2-chat": 1314,
    "gemma-3-27b-it": 1363,
    "gpt-4.1-nano-2025-04-14": 1320,
    "qwen2.5-72b-instruct": 1302,
    "deepseek-r1": 1394,
    "Claude 3.5 Sonnet (10/22)": 1368,
    "llama-4-maverick-17b-128e-instruct": 1327,
    "llama-3.1-nemotron-70b-instruct": 1296,
    "claude-3-7-sonnet-20250219-thinking-32k": 1385,
}

# Mapping from human arena model names to automated evaluation names
humans_to_auto_names: Dict[str, str] = {
    "o3-2025-04-16": "o3-2025-04-16",
    "o4-mini-2025-04-16": "o4-mini-2025-04-16",
    "gemini-2.5-flash": "gemini-2.5-flash",
    "o1-2024-12-17": "o1-2024-12-17",
    "gemma-3-27b-it": "gemma-3-27b-it",
    "o3-mini": "o3-mini-2025-01-31",
    "o3-mini-high": "o3-mini-2025-01-31-high",
    "Qwen3-235B-A22B": "Qwen3-235B-A22B",
    "gpt-4.1-2025-04-14": "gpt-4.1",
    "gpt-4.1-mini-2025-04-14": "gpt-4.1-mini",
    "Qwen3-32B": "Qwen/Qwen3-32B",
    "QwQ-32B": "Qwen/QwQ-32B",
    "Qwen3-30B-A3B": "Qwen/Qwen3-30B-A3B",
    "athene-v2-chat": "Nexusflow/Athene-V2-Chat",
    "gpt-4.1-nano-2025-04-14": "gpt-4.1-nano",
    "qwen2.5-72b-instruct": "Qwen/Qwen2.5-72B-Instruct",
    "gpt-4.5-preview-2025-02-27": "gpt-4.5-preview",
    "Claude 3.5 Sonnet (10/22)": "claude-3-5-sonnet-20241022",
    "deepseek-r1": "deepseek-r1",
    "claude-3-7-sonnet-20250219-thinking-32k": "claude-3-7-sonnet-20250219-thinking-16k",
    "llama-4-maverick-17b-128e-instruct": "llama4-maverick-instruct-basic",
    "llama-3.1-nemotron-70b-instruct": "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF",
}

# Mapping from Artificial Analysis model names to automated evaluation names
artificial_to_auto_names: Dict[str, str] = {
    "Claude 3.5 Sonnet (Oct '24)": "claude-3-5-sonnet-20241022",
    "Claude 3.7 Sonnet (Non-reasoning)": "claude-3-7-sonnet-20250219-thinking-16k",
    "DeepSeek R1 (Jan '25)": "deepseek-r1",
    "GPT-4.1": "gpt-4.1",
    "GPT-4.1 mini": "gpt-4.1-mini",
    "GPT-4.1 nano": "gpt-4.1-nano",
    "GPT-4.5 (Preview)": "gpt-4.5-preview",
    "o3": "o3-2025-04-16",
    "o3-mini": "o3-mini-2025-01-31",
    "o3-mini (high)": "o3-mini-2025-01-31-high",
    "Gemini 2.5 Flash (Non-reasoning)": "gemini-2.5-flash",
    "Gemma 3 27B Instruct": "gemma-3-27b-it",
    "Qwen3 235B A22B (Non-reasoning)": "Qwen3-235B-A22B",
    "Qwen3 30B A3B (Non-reasoning)": "Qwen/Qwen3-30B-A3B",
    "Qwen3 32B (Reasoning)": "Qwen/Qwen3-32B",
    "Qwen2.5 Instruct 72B": "Qwen/Qwen2.5-72B-Instruct",
    "QwQ 32B": "Qwen/QwQ-32B",
    "Llama 3.1 Nemotron Instruct 70B": "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF",
    "Llama 4 Maverick": "llama4-maverick-instruct-basic",
}

# Mapping from automated evaluation names to human-readable pretty names
auto_names_to_pretty_names: Dict[str, str] = {
    "claude-3-5-sonnet-20241022": "Claude 3.5 Sonnet",
    "claude-3-7-sonnet-20250219-thinking-16k": "Claude 3.7 Sonnet thinking 16k",
    "o3-mini-2025-01-31-high": "o3 Mini High",
    "o3-mini-2025-01-31": "o3 Mini",
    "Qwen/QwQ-32B": "QwQ 32B",
    "Qwen/Qwen3-30B-A3B": "Qwen3 30B A3B",
    "Qwen/Qwen3-32B": "Qwen3 32B",
    "o3-2025-04-16": "o3",
    "o4-mini-2025-04-16": "o4 Mini",
    "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF": "Llama 3.1 Nemotron 70B Instruct",
    "Qwen/Qwen2.5-72B-Instruct": "Qwen2.5 72B Instruct",
    "Nexusflow/Athene-V2-Chat": "Athene V2 Chat",
    "llama4-maverick-instruct-basic": "Llama 4 Maverick Instruct",
    "deepseek-r1": "DeepSeek-R1",
    "gpt-4.1-nano": "GPT-4.1 Nano",
    "gpt-4.1-mini": "GPT-4.1 Mini",
    "gpt-4.1": "GPT-4.1",
    "gemma-3-27b-it": "Gemma 3 27B Instruct",
    "gpt-4.5-preview": "GPT-4.5 (Preview)",
    "o1-2024-12-17": "o1",
    "gemini-2.5-flash": "Gemini 2.5 Flash",
    "Qwen3-235B-A22B": "Qwen3 235B A22B",
}

# Mapping from Arena Preference 140 model names to automated evaluation names
arena_pref_140_to_auto: Dict[str, str] = {
    "claude-3-7-sonnet-20250219-thinking-32k": "claude-3-7-sonnet-20250219-thinking-16k",
    "gemini-2.5-flash": "gemini-2.5-flash",
    "gemma-3-27b-it": "gemma-3-27b-it",
    "gpt-4.1-2025-04-14": "gpt-4.1",
    "gpt-4.1-mini-2025-04-14": "gpt-4.1-mini",
    "o3-2025-04-16": "o3-2025-04-16",
    "o3-mini": "o3-mini-2025-01-31",
    "o4-mini-2025-04-16": "o4-mini-2025-04-16",
    "qwen3-235b-a22b": "qwen3-235b-a22b",
    "qwen3-30b-a3b": "qwen/qwen3-30b-a3b",
    "qwq-32b": "qwen/qwq-32b",
    "llama-4-maverick-17b-128e-instruct": "llama4-maverick-instruct-basic",
}

# ============================================================================
# Benchmark Scores
# ============================================================================

# MMLU benchmark scores for various models
mmlu_scores: Dict[str, Optional[float]] = {
    "Qwen3-235B-A22B": 0.6818,
    "Qwen/Qwen3-32B": None,
    "Qwen/QwQ-32B": 0.6907,
    "Qwen/Qwen3-30B-A3B": None,
    "Qwen/Qwen2.5-72B-Instruct": None,
    "gemini-2.5-flash": 0.776,
    "deepseek-r1": 0.84,
    "gpt-4.1": 0.818,
    "gpt-4.1-mini": None,
    "o4-mini-2025-04-16": 0.6309,
    "o1-2024-12-17": 0.893,
    "o3-mini-2025-01-31": 0.794,
}

# ============================================================================
# Pre-computed Rankings
# ============================================================================

# Full matrix rankings for different anchor models
# Each key is an anchor model, value is the ranked list of all models
full_matrix_ranking: Dict[str, List[str]] = {
    "Qwen3-8B": [
        "o3-2025-04-16",
        "Qwen3-235B-A22B",
        "Qwen/Qwen3-32B",
        "gemini-2.5-flash",
        "Qwen/QwQ-32B",
        "deepseek-r1",
        "o4-mini-2025-04-16",
        "Qwen/Qwen3-30B-A3B",
        "gpt-4.1",
        "o3-mini-2025-01-31-high",
        "claude-3-7-sonnet-20250219-thinking-16k",
        "gpt-4.1-mini",
        "o3-mini-2025-01-31",
        "o1-2024-12-17",
        "gpt-4.5-preview",
        "gemma-3-27b-it",
        "Nexusflow/Athene-V2-Chat",
        "gpt-4.1-nano",
        "claude-3-5-sonnet-20241022",
        "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF",
        "Qwen/Qwen2.5-72B-Instruct",
        "llama4-maverick-instruct-basic",
    ],
    "DeepSeek-V3": [
        "o3-2025-04-16",
        "Qwen3-235B-A22B",
        "gemini-2.5-flash",
        "deepseek-r1",
        "Qwen/Qwen3-32B",
        "o4-mini-2025-04-16",
        "gpt-4.1",
        "Qwen/QwQ-32B",
        "claude-3-7-sonnet-20250219-thinking-16k",
        "gpt-4.5-preview",
        "o3-mini-2025-01-31-high",
        "Qwen/Qwen3-30B-A3B",
        "o1-2024-12-17",
        "gemma-3-27b-it",
        "o3-mini-2025-01-31",
        "gpt-4.1-mini",
        "claude-3-5-sonnet-20241022",
        "Nexusflow/Athene-V2-Chat",
        "gpt-4.1-nano",
        "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF",
        "Qwen/Qwen2.5-72B-Instruct",
        "llama4-maverick-instruct-basic",
    ],
    "gpt-oss-120b": [
        "o3-2025-04-16",
        "o4-mini-2025-04-16",
        "gemini-2.5-flash",
        "o3-mini-2025-01-31-high",
        "Qwen3-235B-A22B",
        "gpt-4.1",
        "o3-mini-2025-01-31",
        "o1-2024-12-17",
        "Qwen/Qwen3-32B",
        "gpt-4.1-mini",
        "deepseek-r1",
        "claude-3-7-sonnet-20250219-thinking-16k",
        "Qwen/QwQ-32B",
        "gpt-4.5-preview",
        "Qwen/Qwen3-30B-A3B",
        "gemma-3-27b-it",
        "Nexusflow/Athene-V2-Chat",
        "claude-3-5-sonnet-20241022",
        "gpt-4.1-nano",
        "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF",
        "llama4-maverick-instruct-basic",
        "Qwen/Qwen2.5-72B-Instruct",
    ],
    "Qwen3-235B-A22B-Instruct-2507": [
        "o3-2025-04-16",
        "Qwen3-235B-A22B",
        "gemini-2.5-flash",
        "Qwen/Qwen3-32B",
        "deepseek-r1",
        "Qwen/QwQ-32B",
        "gpt-4.1",
        "o4-mini-2025-04-16",
        "Qwen/Qwen3-30B-A3B",
        "claude-3-7-sonnet-20250219-thinking-16k",
        "o3-mini-2025-01-31-high",
        "gpt-4.5-preview",
        "gpt-4.1-mini",
        "gemma-3-27b-it",
        "o1-2024-12-17",
        "o3-mini-2025-01-31",
        "Nexusflow/Athene-V2-Chat",
        "claude-3-5-sonnet-20241022",
        "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF",
        "gpt-4.1-nano",
        "Qwen/Qwen2.5-72B-Instruct",
        "llama4-maverick-instruct-basic",
    ],
    "gpt-oss-20b": [
        "o3-2025-04-16",
        "gemini-2.5-flash",
        "o4-mini-2025-04-16",
        "Qwen3-235B-A22B",
        "o3-mini-2025-01-31-high",
        "Qwen/Qwen3-32B",
        "gpt-4.1",
        "o3-mini-2025-01-31",
        "gpt-4.1-mini",
        "Qwen/QwQ-32B",
        "deepseek-r1",
        "o1-2024-12-17",
        "Qwen/Qwen3-30B-A3B",
        "claude-3-7-sonnet-20250219-thinking-16k",
        "gpt-4.5-preview",
        "gemma-3-27b-it",
        "Nexusflow/Athene-V2-Chat",
        "gpt-4.1-nano",
        "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF",
        "claude-3-5-sonnet-20241022",
        "Qwen/Qwen2.5-72B-Instruct",
        "llama4-maverick-instruct-basic",
    ],
}

# Models not included in Artificial Analysis benchmark
models_not_in_artificial_analysis: List[str] = [
    "o4-mini-2025-04-16",
    "gpt-4.5-preview",
    "Nexusflow/Athene-V2-Chat",
]

# LiveCodeBench leaderboard
liveCodeBench: List[str] = [
    "o3-2025-04-16",
    "o3-mini-2025-01-31-high",
    "o3-mini-2025-01-31",
    "o1-2024-12-17",
    "Qwen/QwQ-32B",
    "deepseek-r1",
    "gemini-2.5-flash",
    "gpt-4.1-mini",
    "claude-3-7-sonnet-20250219-thinking-16k",
    "gpt-4.1",
    "llama4-maverick-instruct-basic",
    "claude-3-5-sonnet-20241022",
    "Qwen3-235B-A22B",
    "gpt-4.1-nano",
    "Qwen/Qwen3-30B-A3B",
    "Qwen/Qwen3-32B",
    "Qwen/Qwen2.5-72B-Instruct",
    "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF",
    "gemma-3-27b-it",
]

# MMLU leaderboard
MMLU: List[str] = [
    "o3-2025-04-16",
    "deepseek-r1",
    "o1-2024-12-17",
    "claude-3-7-sonnet-20250219-thinking-16k",
    "llama4-maverick-instruct-basic",
    "gemini-2.5-flash",
    "gpt-4.1",
    "o3-mini-2025-01-31-high",
    "o3-mini-2025-01-31",
    "gpt-4.1-mini",
    "claude-3-5-sonnet-20241022",
    "Qwen/QwQ-32B",
    "Qwen3-235B-A22B",
    "Qwen/Qwen3-32B",
    "Qwen/Qwen2.5-72B-Instruct",
    "Qwen/Qwen3-30B-A3B",
    "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF",
    "gemma-3-27b-it",
    "gpt-4.1-nano",
]

# ============================================================================
# Correlation Statistics
# ============================================================================

# Correlation between full-matrix rankings and human preferences for different judges
full_matrix_with_humans_corr: Dict[str, float] = {
    "Qwen3-8B": 0.33333333333333337,
    "DeepSeek-V3": 0.4952380952380953,
    "gpt-oss-120b": 0.5142857142857143,
    "Qwen3-235B-A22B-Instruct-2507": 0.4285714285714286,
    "gpt-oss-20b": 0.4190476190476191,
}

# Correlation between higher-performing anchor-based rankings and full matrix rankings
higher_anchor_based_with_full_matrix_corr: Dict[str, float] = {
    "Qwen3-8B": 0.9567099567099567,
    "DeepSeek-V3": 0.9826839826839827,
    "gpt-oss-120b": 0.9567099567099567,
    "Qwen3-235B-A22B-Instruct-2507": 0.9653679653679654,
    "gpt-oss-20b": 0.974025974025974,
}

# Correlation between lower-performing anchor-based rankings and full matrix rankings
lower_anchor_based_with_full_matrix_corr: Dict[str, float] = {
    "Qwen3-8B": 0.8268398268398268,
    "DeepSeek-V3": 0.8441558441558441,
    "gpt-oss-120b": 0.7662337662337663,
    "Qwen3-235B-A22B-Instruct-2507": 0.8181818181818182,
    "gpt-oss-20b": 0.7835497835497836,
}

# Correlation between higher-performing anchor-based rankings and human preferences
higher_anchor_based_with_humans_corr: Dict[str, float] = {
    "Qwen3-8B": 0.40952380952380957,
    "DeepSeek-V3": 0.5619047619047619,
    "gpt-oss-120b": 0.580952380952381,
    "Qwen3-235B-A22B-Instruct-2507": 0.5047619047619049,
    "gpt-oss-20b": 0.5047619047619049,
}

# Correlation between lower-performing anchor-based rankings and human preferences
lower_anchor_based_with_humans_corr: Dict[str, float] = {
    "Qwen3-8B": 0.2285714285714286,
    "DeepSeek-V3": 0.37142857142857144,
    "gpt-oss-120b": 0.37142857142857144,
    "Qwen3-235B-A22B-Instruct-2507": 0.2761904761904762,
    "gpt-oss-20b": 0.2,
}

