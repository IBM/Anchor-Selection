# LMJ Anchor Selection
This is the code for the paper ["Mediocrity is the key for LLM as a Judge Anchor Selection"](https://arxiv.org/abs/2603.16848)

## Paper Abstract 
> The "LLM-as-a-judge" paradigm has become a standard method for evaluating open-ended generation. To address the quadratic scalability costs of pairwise comparisons, popular benchmarks like Arena-Hard and AlpacaEval compare all models against a single anchor. However, despite its widespread use, the impact of anchor selection on the reliability of the results remains largely unexplored. In this work, we systematically investigate the effect of anchor selection by evaluating 22 different anchors on the Arena-Hard-v2.0 dataset. We find that the choice of anchor is critical: a poor anchor can dramatically reduce correlation with human rankings. We identify that common anchor choices (best-performing and worst-performing models) make poor anchors. Because these extreme anchors are consistently better or worse than all other models, they are seldom indicative of the relative ranking of the models. We further quantify the effect size of anchor selection, showing it is comparable to the selection of a judge model. We conclude with actionable recommendations. First, we conduct a power analysis, and compute sufficient benchmark sizes for anchor-based evaluation, finding that standard benchmark sizes are insufficient for pairwise evaluation and fail to distinguish between competitive models reliably. Second, we provide guidelines for selecting informative anchors to ensure reliable and efficient evaluation practices.

## 📊 Judgements Dataset

The judgements data from our paper experiments (900K+ evaluations) is available on HuggingFace:

🤗 **[ibm-research/900K-Judgements](https://huggingface.co/datasets/ibm-research/900K-Judgements)**

This dataset contains all pairwise judgements we collected across multiple judge models and benchmarks for the paper's analysis.

## 🎯 Overview

- Generate LLM-as-a-judge (LMJ) judgmenents between multiple LLM models
- Simulate different anchor choices 
- Compute Bradley-Terry rankings and win-rate matrices
- Measure anchor ranking correlations with quadratic/human ranking
- Support for Arena Hard and AlpacaEval datasets

## 📋 Table of Contents

- [Judgements Dataset](#judgements-dataset)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core Scripts](#core-scripts)
- [Data Formats](#data-formats)

## 🚀 Installation

### Prerequisites

- Python 3.8+
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/IBM/Anchor-Selection.git
cd Anchor-Selection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment variables:
Create a `.env` file in the project root with the appropriate credentials for your provider:

**For OpenAI (default):**
```bash
OPENAI_API_KEY=your_openai_api_key
```

**For Together AI:**
```bash
TOGETHER_API_KEY=your_together_api_key
```

**For OpenRouter:**
```bash
OPENROUTER_API_KEY=your_openrouter_api_key
```

**For Custom OpenAI-compatible endpoint:**
```bash
CUSTOM_API_KEY=your_api_key
CUSTOM_BASE_URL=https://your-api-endpoint.com/v1
```

## 🎬 Quick Start

### Generate Judges Matrix

Run pairwise evaluations between models using different providers:

**Using OpenAI (default):**
```bash
python async_run_judges.py \
  --data_path ./model_answers \
  --model_name gpt-4 \
  --models_list model1 model2 model3
```

**Using Together AI:**
```bash
python async_run_judges.py \
  --data_path ./model_answers \
  --model_name meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo \
  --provider together \
  --models_list model1 model2 model3
```

**Using OpenRouter:**
```bash
python async_run_judges.py \
  --data_path ./model_answers \
  --model_name anthropic/claude-3.5-sonnet \
  --provider openrouter \
  --models_list model1 model2 model3
```

### Continue Previous Run

Resume an interrupted evaluation:

```bash
python async_run_judges.py \
  --data_path ./model_answers \
  --model_name gpt-4 \
  --models_list model1 model2 model3 \
  --continue_exp \
  --output_dir ./previous_results
```

### Merge Results

Combine individual evaluation files:

```bash
python merge_eval_files.py --input_dir ./eval_directory
```

## 📚 Core Scripts

### `async_run_judges.py`

Script for generating pairwise model evaluations.

**Key Features:**
- Asynchronous batch processing for efficiency
- Randomized response ordering to reduce position bias
- Progress tracking with ETA
- Support for Arena Hard and AlpacaEval datasets
- Automatic retry on failures

**Arguments:**
- `--data_path`: Path to model response data
- `--model_name`: Judge model identifier
- `--provider`: API provider - `openai` (default), `together`, `openrouter`, or `custom`
- `--models_list`: Space-separated list of models to evaluate
- `--output_dir`: Directory for results (auto-generated if not specified)
- `--continue_exp`: Resume previous run
- `--num_examples`: Limit number of examples (default: all)
- `--max_concurrent`: Concurrent evaluations (default: 3)
- `--temperature`: Sampling temperature (default: 0)
- `--max_tokens`: Max tokens per response (default: 1024)
- `--alpaca_eval_path`: Path to AlpacaEval format data
- `--no_progress`: Disable progress bar

**Examples:**

Using OpenAI:
```bash
python async_run_judges.py \
  --data_path ./arena_hard_data \
  --model_name gpt-4o \
  --provider openai \
  --models_list Qwen3-235B-A22B gpt-4.1 gpt-4.1-mini \
  --max_concurrent 5 \
  --num_examples 100
```

Using Together AI:
```bash
python async_run_judges.py \
  --data_path ./arena_hard_data \
  --model_name meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo \
  --provider together \
  --models_list model1 model2 model3 \
  --max_concurrent 5
```

### `correlations_exp.py`

Main script for comprehensive analysis for evaluation results.

**Features:**
- Kendall's Tau correlation with human preferences
- Bradley-Terry model rankings
- Win rate matrices
- Anchor-based ranking analysis
- Statistical significance testing


## 📊 Data Formats

### Evaluation Output Format

```json
{
  "uid": "eval_id",
  "instruction": "user prompt",
  "model_1_name": "model_a",
  "model_2_name": "model_b",
  "model_1_response": "response from model a",
  "model_2_response": "response from model b",
  "final_verdict": "model_a>model_b",
  "confidence": "slightly",
  "judge_model": "gpt-4",
  "timestamp": "2026-01-29T07:52:01.199Z"
}
```



## 📝 Citation

If you use this framework in your research, please cite:

```
@misc{donyehiya2026mediocritykeyllmjudge,
      title={Mediocrity is the key for LLM as a Judge Anchor Selection}, 
      author={Shachar Don-Yehiya and Asaf Yehudai and Leshem Choshen and Omri Abend},
      year={2026},
      eprint={2603.16848},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2603.16848}, 
}
```

## 📄 License

This project is licensed under the Apache-2.0 License - see the [LICENSE](LICENSE) file for details.
