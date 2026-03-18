"""
Asynchronous Judge Evaluation

This module provides an asynchronous framework for running pairwise evaluations of LLM responses
using judge models.
Key Features:
- Asynchronous evaluation of model response pairs
- Randomized response ordering to mitigate position bias
- Progress tracking with ETA estimation
- Support for Arena Hard and AlpacaEval data formats
- Concurrent request handling with configurable limits
- Comprehensive result logging and error handling

Usage:
    python async_run_judges.py --model_name <judge_model> --data_path <path_to_data>
    
Example:
    python async_run_judges.py --model_name meta-llama/llama-3-3-70b-instruct \
        --data_path ./arena-hard-data --max_concurrent 5
"""

import os
import asyncio
import aiofiles
from dotenv import load_dotenv
from typing import Any, List, Dict
from openai import AsyncOpenAI
from datetime import datetime, timedelta
import json
import argparse
from enum import Enum
from typing import Tuple, Any
import random
import re
import time
from tqdm.asyncio import tqdm

# Evaluation prompt template for pairwise comparison
# Uses a 5-point scale from significantly better to significantly worse
EVALUATION_PROMPT = ("Here is a user input and responses from two assistants, A and B. Which response is better?\n"
                    "You must output only one of the following choices as your final verdict with a label:\n\n"
                    "1. Assistant A is significantly better: [[A>>B]]\n"
                    "2. Assistant A is slightly better: [[A>B]]\n"
                    "3. Tie, relatively the same: [[A=B]]\n"
                    "4. Assistant B is slightly better: [[B>A]]\n"
                    "5. Assistant B is significantly better: [[B>>A]]\n\n"
                    "Example output: \"My final verdict is tie: [[A=B]]\".\n\n"
                    "<|User Prompt|>\n"
                    "{instruction}\n\n"
                    "<|The Start of Assistant A's Answer|>\n"
                    "{response_a}\n"
                    "<|The End of Assistant A's Answer|>\n\n"
                    "<|The Start of Assistant B's Answer|>\n"
                    "{response_b}\n"
                    "<|The End of Assistant B's Answer|>\n"
                    "Final Verdict:"
                    )
 
class ProgressTracker:
    """Track progress of batch evaluations with statistics and ETA"""

    def __init__(self, total_tasks: int, description: str = "Processing"):
        self.total_tasks = total_tasks
        self.completed_tasks = 0
        self.failed_tasks = 0
        self.start_time = time.time()
        self.description = description
        self.last_update_time = self.start_time

    def update(self, success: bool = True):
        """Update progress counters"""
        if success:
            self.completed_tasks += 1
        else:
            self.failed_tasks += 1

        self.last_update_time = time.time()

    def get_progress_stats(self) -> Dict[str, Any]:
        """Get current progress statistics"""
        elapsed_time = time.time() - self.start_time
        total_processed = self.completed_tasks + self.failed_tasks

        if total_processed == 0:
            return {
                'completed': self.completed_tasks,
                'failed': self.failed_tasks,
                'total': self.total_tasks,
                'progress_pct': 0.0,
                'elapsed_time': elapsed_time,
                'eta_seconds': None,
                'rate_per_second': 0.0
            }

        progress_pct = (total_processed / self.total_tasks) * 100
        rate_per_second = total_processed / elapsed_time if elapsed_time > 0 else 0

        # Calculate ETA
        remaining_tasks = self.total_tasks - total_processed
        eta_seconds = remaining_tasks / rate_per_second if rate_per_second > 0 else None

        return {
            'completed': self.completed_tasks,
            'failed': self.failed_tasks,
            'total': self.total_tasks,
            'progress_pct': progress_pct,
            'elapsed_time': elapsed_time,
            'eta_seconds': eta_seconds,
            'rate_per_second': rate_per_second
        }

    def format_eta(self, eta_seconds: float) -> str:
        """Format ETA as human-readable string"""
        if eta_seconds is None:
            return "Unknown"

        eta_timedelta = timedelta(seconds=int(eta_seconds))
        return str(eta_timedelta)

    
    def print_progress(self):
        """Print current progress status"""
        stats = self.get_progress_stats()
        eta_str = self.format_eta(stats['eta_seconds'])

        print(f"\r{self.description}: {stats['completed']}/{stats['total']} completed "
              f"({stats['progress_pct']:.1f}%) | "
              f"Failed: {stats['failed']} | "
              f"Rate: {stats['rate_per_second']:.2f}/s | "
              f"ETA: {eta_str}", end='', flush=True)
        

def init_model(model_name: str, temperature: float = 0, max_tokens: int = 2048, provider: str = "openai") -> AsyncOpenAI:
    """
    Initialize an async OpenAI-compatible client for various providers.
    
    Args:
        model_name: Name of the model to use
        temperature: Sampling temperature
        max_tokens: Maximum tokens in response
        provider: Provider name - "openai", "anthropic", "together", "openrouter", or "custom"
    
    Returns:
        AsyncOpenAI client instance
    """
    load_dotenv()
    
    if provider == "openai":
        # OpenAI official API
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY must be set in environment variables")
        
        client = AsyncOpenAI(
            api_key=api_key,
            max_retries=2
        )
        print(f"Initialized OpenAI client")
        
    elif provider == "anthropic":
        # Anthropic via OpenAI-compatible endpoint
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY must be set in environment variables")
        
        # Note: Anthropic doesn't have a direct OpenAI-compatible API
        # This is a placeholder - you may need to use the anthropic library instead
        print("Warning: Anthropic requires the anthropic library, not OpenAI client")
        print("Consider using the anthropic async client instead")
        client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.anthropic.com/v1",  # Placeholder
            max_retries=2
        )
        
    elif provider == "together":
        # Together AI
        api_key = os.getenv("TOGETHER_API_KEY")
        if not api_key:
            raise ValueError("TOGETHER_API_KEY must be set in environment variables")
        
        client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.together.xyz/v1",
            max_retries=2
        )
        print(f"Initialized Together AI client")
        
    elif provider == "openrouter":
        # OpenRouter
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY must be set in environment variables")
        
        client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            max_retries=2
        )
        print(f"Initialized OpenRouter client")
        
    elif provider == "custom":
        # Custom OpenAI-compatible endpoint
        api_key = os.getenv("CUSTOM_API_KEY")
        base_url = os.getenv("CUSTOM_BASE_URL")
        if not api_key or not base_url:
            raise ValueError("CUSTOM_API_KEY and CUSTOM_BASE_URL must be set in environment variables")
        
        client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            max_retries=2
        )
        print(f"Initialized custom client - base_url: {base_url}")
        
    else:
        raise ValueError(f"Unknown provider: {provider}. Supported providers: openai, anthropic, together, openrouter, custom")
    
    # Store model config on the client for easy access
    client.model_name = model_name
    client.temperature = temperature
    client.max_tokens = max_tokens
    client.provider = provider
    
    print(f"Model: {model_name} | Provider: {provider}")
    return client


async def run_prompt_async(client: AsyncOpenAI, prompt: str, prompt_id: str = None, output_dir: str = "outputs") -> str:
    """
    Run a single prompt asynchronously and save output to file
    
    Args:
        client: AsyncOpenAI client instance
        prompt: The prompt text
        prompt_id: Optional identifier for the prompt (used in filename)
        output_dir: Directory to save outputs
    
    Returns:
        The response content
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Count input tokens (you may need to implement InputTokensCounter)
        # InputTokensCounter().add(len(prompt))
        
        # Use OpenAI async completion
        response = await client.chat.completions.create(
            model=client.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=client.temperature,
            max_tokens=client.max_tokens
        )
        
        # Extract the response content
        response_content = response.choices[0].message.content
        
        # Count output tokens (you may need to implement OutputTokensCounter)
        # OutputTokensCounter().add(len(response_content))
        
        # Generate filename
        if prompt_id:
            filename = f"{prompt_id}.txt"
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"response_{timestamp}.txt"
        
        filepath = os.path.join(output_dir, filename)
        
        # Save output to file
        async with aiofiles.open(filepath, 'w', encoding='utf-8') as f:
            await f.write(f"Prompt: {prompt}\n")
            await f.write(f"{'='*50}\n")
            await f.write(f"Response: {response_content}\n")
            await f.write(f"{'='*50}\n")
            await f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            await f.write(f"Model: {client.model_name}\n")
            await f.write(f"Temperature: {client.temperature}\n")
            await f.write(f"Max Tokens: {client.max_tokens}\n")
            
            # Add token usage info if available
            if hasattr(response, 'usage') and response.usage:
                await f.write(f"Input Tokens: {response.usage.prompt_tokens}\n")
                await f.write(f"Output Tokens: {response.usage.completion_tokens}\n")
                await f.write(f"Total Tokens: {response.usage.total_tokens}\n")
        
        print(f"Response saved to: {filepath}")
        print(f"Response preview: {response_content[:100]}...")
        
        return response_content
        
    except Exception as e:
        error_msg = f"Error processing prompt: {str(e)}"
        print(error_msg)
        
        # Generate timestamp for error file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        
        # Save error to file
        error_filename = f"error_{prompt_id}_{timestamp}.txt" if prompt_id else f"error_{timestamp}.txt"
        error_filepath = os.path.join(output_dir, error_filename)
        
        async with aiofiles.open(error_filepath, 'w', encoding='utf-8') as f:
            await f.write(f"Error: {error_msg}\n")
            await f.write(f"Prompt: {prompt}\n")
            await f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            await f.write(f"Model: {client.model_name}\n")
        
        return error_msg
    

def extract_verdict(judgment: str) -> Tuple[str, str]:
    """
    Extract the verdict from judge response
    
    Returns:
        Tuple of (verdict_label, confidence)
        verdict_label: A>>B, A>B, A=B, B>A, B>>A, or INVALID
        confidence: significantly, slightly, tie, or unknown
    """
    # Look for verdict patterns
    patterns = [
        r'\[\[A>>B\]\]',  # A significantly better
        r'\[\[A>B\]\]',   # A slightly better
        r'\[\[A=B\]\]',   # Tie
        r'\[\[B>A\]\]',   # B slightly better
        r'\[\[B>>A\]\]'   # B significantly better
    ]
    
    for pattern in patterns:
        if re.search(pattern, judgment, re.IGNORECASE):
            verdict = pattern.strip('r\\[]')
            if '>>' in verdict:
                confidence = 'significantly'
            elif '>' in verdict or '<' in verdict:
                confidence = 'slightly'
            elif '=' in verdict:
                confidence = 'tie'
            else:
                confidence = 'unknown'
            return verdict, confidence
    
    return 'INVALID', 'unknown'

async def run_pairwise_evaluation(
    judge_client: AsyncOpenAI, 
    instruction: str, 
    response_1: str, 
    response_2: str, 
    model_1_name: str = "Model1",
    model_2_name: str = "Model2",
    uid: str = None, 
    output_dir: str = "outputs"
) -> Dict[str, Any]:
    """
    Run pairwise evaluation with randomized order
    
    Args:
        judge_client: AsyncOpenAI client for judge model
        instruction: User instruction/prompt
        response_1: Response from first model
        response_2: Response from second model
        model_1_name: Name of first model
        model_2_name: Name of second model
        uid: Optional identifier for evaluation
        output_dir: Directory to save evaluation results
    
    Returns:
        Dictionary containing evaluation results
    """
    try:
        if not os.path.isdir(output_dir):
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
        
        # Randomize order (50% chance to swap)
        swap_order = random.choice([True, False])
        
        if swap_order:
            response_a, response_b = response_2, response_1
            model_a_name, model_b_name = model_2_name, model_1_name
            original_order = "swapped"
        else:
            response_a, response_b = response_1, response_2
            model_a_name, model_b_name = model_1_name, model_2_name
            original_order = "original"
        
        # Format evaluation prompt
        evaluation_prompt = EVALUATION_PROMPT.format(
            instruction=instruction,
            response_a=response_a,
            response_b=response_b
        )

        if judge_client.model_name == "Qwen/Qwen3-8B":
            print("model is", judge_client.model_name)
            if len(evaluation_prompt) > (40960 - 160):
                print("making it shorter!")
                num_tokens = (40960 - 160) - len(evaluation_prompt)
                num_tokens_per_response = int(num_tokens / 2)
                evaluation_prompt = EVALUATION_PROMPT.format(
                    instruction=instruction,
                    response_a=response_a[:num_tokens_per_response],
                    response_b=response_b[:num_tokens_per_response]
                )
            else:
                print("len is ok", len(evaluation_prompt))            
        
        # Get judgment
        response = await judge_client.chat.completions.create(
            model=judge_client.model_name,
            messages=[{"role": "user", "content": evaluation_prompt}],
            temperature=judge_client.temperature,
            max_tokens=judge_client.max_tokens
        )
        
        judgment = response.choices[0].message.content
        print(judgment)
        
        # Extract verdict
        verdict_label, confidence = extract_verdict(judgment)
        
        # Map verdict back to original models
        if swap_order:
            # If we swapped, need to flip the verdict
            verdict_mapping = {
                'A>>B': f'{model_2_name}>>{model_1_name}',
                'A>B': f'{model_2_name}>{model_1_name}',
                'A=B': f'{model_1_name}={model_2_name}',
                'B>A': f'{model_1_name}>{model_2_name}',
                'B>>A': f'{model_1_name}>>{model_2_name}',
                'INVALID': 'INVALID'
            }
        else:
            verdict_mapping = {
                'A>>B': f'{model_1_name}>>{model_2_name}',
                'A>B': f'{model_1_name}>{model_2_name}',
                'A=B': f'{model_1_name}={model_2_name}',
                'B>A': f'{model_2_name}>{model_1_name}',
                'B>>A': f'{model_2_name}>>{model_1_name}',
                'INVALID': 'INVALID'
            }
        
        final_verdict = verdict_mapping.get(verdict_label, 'INVALID')
        
        # Prepare results
        results = {
            'uid': uid,
            'instruction': instruction,
            'model_1_name': model_1_name,
            'model_2_name': model_2_name,
            'model_1_response': response_1,
            'model_2_response': response_2,
            'order_randomized': original_order,
            'model_a_in_prompt': model_a_name,
            'model_b_in_prompt': model_b_name,
            'raw_judgment': judgment,
            'extracted_verdict': verdict_label,
            'confidence': confidence,
            'final_verdict': final_verdict,
            'timestamp': datetime.now().isoformat(),
            'judge_model': judge_client.model_name
        }
        
        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        if uid:
            filename = f"eval_{uid}.json"
        else:
            filename = f"eval_{timestamp}.json"
        
        filepath = os.path.join(output_dir, filename)
        
        async with aiofiles.open(filepath, 'w', encoding='utf-8') as f:
            await f.write(json.dumps(results, indent=2, ensure_ascii=False))
        
        print(f"Evaluation saved to: {filepath}")
        print(f"Verdict: {final_verdict}")
        
        return results
        
    except Exception as e:
        error_msg = f"Error in pairwise evaluation: {str(e)}"
        print(error_msg)
        return {'error': error_msg, 'uid': uid}

async def run_batch_pairwise_evaluation(
    judge_client: AsyncOpenAI,
    evaluation_data: List[Dict[str, Any]],
    output_dir: str = "outputs",
    max_concurrent: int = 3,
    show_progress: bool = True
) -> List[Dict[str, Any]]:
    """
    Run batch pairwise evaluations
    
    Args:
        judge_client: AsyncOpenAI client for judge model
        evaluation_data: List of evaluation dictionaries
        output_dir: Directory to save results
        max_concurrent: Maximum concurrent evaluations
    
    Expected format for evaluation_data:
    [
        {
            "instruction": "User instruction",
            "response_1": "Response from model 1",
            "response_2": "Response from model 2",
            "model_1_name": "GPT-4",  # optional
            "model_2_name": "Claude",  # optional
            "uid": "eval_001"  # optional
        }
    ]
    
    Returns:
        List of evaluation results
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def evaluate_with_semaphore(eval_item, pbar=None):
        async with semaphore:
            try:
                result = await run_pairwise_evaluation(
                    judge_client=judge_client,
                    instruction=eval_item['instruction'],
                    response_1=eval_item['response_1'],
                    response_2=eval_item['response_2'],
                    model_1_name=eval_item.get('model_1_name', 'Model1'),
                    model_2_name=eval_item.get('model_2_name', 'Model2'),
                    uid=eval_item.get('uid'),
                    output_dir=output_dir
                )
                if pbar:
                    pbar.update(1)
                return result
            except Exception as e:
                if pbar:
                    pbar.update(1)
                return {'error': str(e), 'uid': eval_item.get('uid')}
    
    if show_progress:
        # Use tqdm for progress tracking
        pbar = tqdm(total=len(evaluation_data), desc="Evaluating pairs", unit="pair")
        tasks = [evaluate_with_semaphore(item, pbar) for item in evaluation_data]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        pbar.close()
    else:
        # Run without progress bar
        tasks = [evaluate_with_semaphore(item) for item in evaluation_data]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Save summary results
    summary_file = os.path.join(output_dir, f"batch_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    async with aiofiles.open(summary_file, 'w', encoding='utf-8') as f:
        await f.write(json.dumps(results, indent=2, ensure_ascii=False))
    
    return results

async def run_multiple_prompts(client: AsyncOpenAI, prompts: List[Dict[str, str]], output_dir: str = "outputs", max_concurrent: int = 5) -> List[str]:
    """
    Run multiple prompts concurrently
    
    Args:
        client: AsyncOpenAI client instance
        prompts: List of dictionaries with 'prompt' and optional 'id' keys
        output_dir: Directory to save outputs
        max_concurrent: Maximum number of concurrent requests
    
    Returns:
        List of response contents
    """
    # Create semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def run_with_semaphore(prompt_data):
        async with semaphore:
            prompt_text = prompt_data.get('prompt', '')
            prompt_id = prompt_data.get('id', None)
            return await run_prompt_async(client, prompt_text, prompt_id, output_dir)
    
    # Run all prompts concurrently
    tasks = [run_with_semaphore(prompt_data) for prompt_data in prompts]
    responses = await asyncio.gather(*tasks, return_exceptions=True)
    
    return responses

async def run_prompts_from_file(client: AsyncOpenAI, file_path: str, output_dir: str = "outputs", max_concurrent: int = 5) -> List[str]:
    """
    Run prompts from a JSON file
    
    Args:
        client: AsyncOpenAI client instance
        file_path: Path to JSON file containing prompts
        output_dir: Directory to save outputs
        max_concurrent: Maximum number of concurrent requests
    
    Returns:
        List of response contents
    
    Expected JSON format:
    [
        {"prompt": "Question 1", "id": "q1"},
        {"prompt": "Question 2", "id": "q2"}
    ]
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            prompts = json.load(f)
        
        return await run_multiple_prompts(client, prompts, output_dir, max_concurrent)
    
    except Exception as e:
        print(f"Error reading prompts file: {e}")
        return []

# Example usage
async def run_exp(model_name: str, evaluation_data: List[dict], temperature: float = 0, max_tokens: int = 2048, output_dir: str = "outputs", show_progress: bool = True, provider: str = "openai"):
    # Initialize judge model
    judge_client = init_model(model_name, temperature, max_tokens, provider)
    
    print(f"\nRunning batch evaluation with {len(evaluation_data)} pairs...")
    print(f"Judge model: {model_name}")
    print(f"Output directory: {output_dir}")
    print(f"Progress tracking: {'Enabled' if show_progress else 'Disabled'}")
    
    # Initialize progress tracker
    progress_tracker = ProgressTracker(len(evaluation_data), "Evaluating pairs")
    
    # Start timing
    start_time = time.time()
    
    batch_results = await run_batch_pairwise_evaluation(
        judge_client=judge_client,
        evaluation_data=evaluation_data,
        output_dir=output_dir,
        max_concurrent=3,
        show_progress=show_progress
    )
    
    # Calculate final statistics
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\n\nCompleted {len(batch_results)} evaluations in {total_time:.2f} seconds")
    
    # Print detailed summary
    valid_results = [r for r in batch_results if not isinstance(r, Exception) and 'error' not in r]
    error_results = [r for r in batch_results if isinstance(r, Exception) or 'error' in r]
    
    print(f"Successful evaluations: {len(valid_results)}")
    print(f"Failed evaluations: {len(error_results)}")
    print(f"Success rate: {len(valid_results)/len(batch_results)*100:.1f}%")
    print(f"Average time per evaluation: {total_time/len(batch_results):.2f} seconds")
    
    if valid_results:
        print("\nSample results:")
        for i, result in enumerate(valid_results[:5]):  # Show first 5 results
            print(f"  {result['uid']}: {result['final_verdict']}")
        if len(valid_results) > 5:
            print(f"  ... and {len(valid_results) - 5} more")
    
    if error_results:
        print(f"\nErrors encountered: {len(error_results)}")
        for i, error in enumerate(error_results[:3]):  # Show first 3 errors
            if isinstance(error, Exception):
                print(f"  Exception: {str(error)}")
            else:
                print(f"  {error.get('uid', 'Unknown')}: {error.get('error', 'Unknown error')}")
        if len(error_results) > 3:
            print(f"  ... and {len(error_results) - 3} more errors")
    
    # Close the client
    await judge_client.close()
    
    return batch_results


def load_arena_hard_responses(path):
    uid_to_model = {}
    model_names = []
    # Iterate over all files in the directory
    for filename in os.listdir(path):
        if filename.endswith('.jsonl'):
            filepath = os.path.join(path, filename)
            print(f"Processing file: {filename}")
            
            with open(filepath, 'r', encoding='utf-8') as file:
                for line_number, line in enumerate(file, start=1):
                    try:
                        data = json.loads(line)
                        # print(data)
                        if "uid" in data:
                            if data["uid"] not in uid_to_model:
                                uid_to_model[data["uid"]] = {data["model"]: data}
                            else:
                                uid_to_model[data["uid"]][data["model"]] = data
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON on line {line_number} in {filename}: {e}")
            # model_names.append(filename[:-len('.jsonl')])
            model_names.append(data["model"])


    # print(uid_to_model.keys())
    return uid_to_model, model_names
    

def load_new_dataset_responses(path):
    """
    Loads data from the new dataset format where each file is a JSON list
    of responses for a single model.
    
    Assumes that the Nth record in each file corresponds to the same instruction.
    Uses the record index (0, 1, 2, ...) as the effective 'uid'.
    """
    # This dict will store data as: index -> model_name -> record
    index_to_model = {}
    model_names = []
    
    # Iterate over all files in the directory
    for filename in os.listdir(path):
        if filename.endswith('.json'):
            filepath = os.path.join(path, filename)
            print(f"Processing new dataset file: {filename}")
            
            # Derive model name from filename (e.g., 'gpt4.json' -> 'gpt4')
            model_name = filename[:-len('.json')]
            if not model_name:
                continue # Skip empty or invalid filenames
                
            model_names.append(model_name)
            
            try:
                with open(filepath, 'r', encoding='utf-8') as file:
                    # Load the entire JSON list from the file
                    data_list = json.load(file) 
                    
                    for index, record in enumerate(data_list):
                        if index not in index_to_model:
                            index_to_model[index] = {}
                            
                        # Ensure the 'generator' field is present, falling back to model_name
                        if 'generator' not in record:
                            record['generator'] = model_name
                            
                        # Store the entire record, keyed by index, then model_name
                        index_to_model[index][model_name] = record
                        
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON in {filename}: {e}")
            except Exception as e:
                print(f"An error occurred processing {filename}: {e}")
                
    return index_to_model, model_names

def randomize_response_order(response_a: str, response_b: str) -> Tuple[str, str, bool]:
    """
    Randomly flips the order of two responses to avoid order bias.

    Args:
        response_a (str): First model's response.
        response_b (str): Second model's response.

    Returns:
        tuple: (response_1, response_2, flipped)
            - response_1: First response to show
            - response_2: Second response to show
            - flipped: Boolean indicating if the original order was flipped
    """
    if random.random() < 0.5:
        return response_a, response_b, False
    else:
        return response_b, response_a, True

def check_if_valid(item):
    verdict = item.get("final_verdict")
    if verdict == "INVALID":
        return False
    return True


# Run the async main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/llama-3-3-70b-instruct")
    parser.add_argument("--data_path", type=str, default="/Users/shachardon/repo/arena-hard-data/data/arena-hard-v2.0/model_answer")
    parser.add_argument("--provider", type=str, default="openai",
                        choices=["openai", "together", "openrouter", "custom"],
                        help="API provider: openai (default), together, openrouter, or custom")
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--temprature", type=float, default=0)
    parser.add_argument("--models_list", nargs='+', default=None)
    parser.add_argument("--continue_exp", action='store_true')
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--show_models_names", action='store_true')
    parser.add_argument("--num_examples", type=int, default=-1)
    parser.add_argument("--no_progress", action='store_true', help="Disable progress bar")
    parser.add_argument("--max_concurrent", type=int, default=3, help="Maximum concurrent evaluations")
    parser.add_argument("--alpaca_eval_path", type=str, default=None)
    
    args = parser.parse_args()

    if args.continue_exp and args.output_dir:
        output_dir = args.output_dir
        # prepare a list of all existing results
        existing_results = set()
        for filename in os.listdir(output_dir):
            if filename.endswith('.json') and filename.startswith("eval"):
                with open(os.path.join(args.output_dir, filename), "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if check_if_valid(data):
                        existing_results.add(filename[:-len('.json')])
            if filename.endswith('.json') and filename.startswith("merged_eval"):
                with open(os.path.join(args.output_dir, filename), "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for item in data:
                        if check_if_valid(item):
                            existing_results.add(f"eval_{item.get('uid')}")
        # print(existing_results)

    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        output_dir = f"output_{timestamp}"
        os.mkdir(output_dir)    
    
    data_to_evaluate = []
    if args.alpaca_eval_path is not None:
        print("\n--- Loading New Dataset ---")
        # Call the new loading function
        index_to_model, model_names_new = load_new_dataset_responses(args.alpaca_eval_path)

        if args.show_models_names:
            for model_name in model_names_new:
                print(model_name, end=" ")
            print("\n", len(model_names_new))
            exit(0)
        if args.models_list:
            model_names_new = sorted(args.models_list)
        print(model_names_new)
        
        if args.num_examples > 0:
            num_examples = args.num_examples
        else:
            num_examples = len(index_to_model)
        print("number or examples:", num_examples)

        new_data_counter = 0
        for index in index_to_model.keys():
            new_data_counter += 1
            if args.num_examples > 0 and new_data_counter > num_examples:
                break
            
            # Check if we have at least 2 models for this index
            if len(index_to_model[index]) < 2:
                print(f"Skipping index {index}: only found {len(index_to_model[index])} model(s).")
                continue
                
            for i in range(len(model_names_new)):
                for j in range(i + 1, len(model_names_new)):
                    model_a_name = model_names_new[i]
                    model_b_name = model_names_new[j]
                    
                    # Ensure both models have data for this index
                    if model_a_name not in index_to_model[index] or model_b_name not in index_to_model[index]:
                        # This happens if one file has fewer lines than another
                        continue
                        
                    model_a_data = index_to_model[index][model_a_name]
                    model_b_data = index_to_model[index][model_b_name]
                    
                    # Sanity check: instructions should ideally match
                    if model_a_data['instruction'] != model_b_data['instruction']:
                        print(f"Warning: Mismatch in instruction for index {index} between {model_a_name} and {model_b_name}.")
                        # You might want to skip these pairs
                        continue 
                    
                    # Construct the unique ID
                    # Using 'newset' as a prefix to distinguish from the arena_hard uids
                    uid_str = f'eval_newset_{index}_{model_a_name.replace("/", "_")}_{model_b_name.replace("/", "_")}'
                    
                    if args.continue_exp and uid_str in existing_results:
                        print(f'{uid_str} already exists, skipping.')
                        continue

                    # Map new data fields to the target format
                    data = {"instruction": model_a_data['instruction'],
                            "response_1": model_a_data['output'],
                            "response_2": model_b_data['output'],
                            "model_1_name": model_a_data['generator'], # Use the 'generator' field
                            "model_2_name": model_b_data['generator'],
                            "uid": uid_str
                            }
                    data_to_evaluate.append(data)

                print(f"Added new pairs. Total pairs to evaluate: {len(data_to_evaluate)}.")
            else:
                print("\nNo new_data_path specified, skipping new dataset.")

            print(f"\nTotal items in data_to_evaluate: {len(data_to_evaluate)}")

    else:
        uid_to_model, model_names = load_arena_hard_responses(args.data_path)
        if args.show_models_names:
            for model_name in model_names:
                print(model_name, end=" ")
            print("\n", len(model_names))
            exit(0)

        if args.models_list:
            model_names = sorted(args.models_list)
        print(model_names)

        if args.num_examples > 0:
            num_examples = args.num_examples
        else:
            num_examples = len(uid_to_model)
        print("number or examples:", num_examples)

        counter = 0
        for uid in uid_to_model.keys():
            counter += 1
            if args.num_examples > 0 and counter > num_examples:
                break
            for i in range(len(model_names)):
                for j in range(i + 1, len(model_names)):
                    model_a_data = uid_to_model[uid][model_names[i]]
                    model_b_data = uid_to_model[uid][model_names[j]]
                    model_a_name = model_a_data['model']
                    model_b_name = model_b_data['model']
                    if args.continue_exp: # skip data points that we already have
                        if f'eval_{uid}_{model_names[i].replace("/", "_")}_{model_names[j].replace("/", "_")}' in existing_results:
                            print(f'eval_{uid}_{model_names[i].replace("/", "_")}_{model_names[j].replace("/", "_")} already exists in the output dir')
                            continue
                    data = {"instruction": model_a_data['messages'][0]['content'],
                            "response_1": model_a_data['messages'][1]['content']['answer'],
                            "response_2": model_b_data['messages'][1]['content']['answer'],
                            "model_1_name": model_a_name,
                            "model_2_name": model_b_name,
                            "uid": f'{uid}_{model_names[i].replace("/", "_")}_{model_names[j].replace("/", "_")}'
                            }
                    data_to_evaluate.append(data)
        print("len data_to_evaluate", len(data_to_evaluate))
    # exit(0)

    # Run the experiment with progress tracking
    show_progress = not args.no_progress
    asyncio.run(run_exp(
        model_name=args.model_name,
        evaluation_data=data_to_evaluate,
        temperature=args.temprature,
        max_tokens=args.max_tokens,
        output_dir=output_dir,
        show_progress=show_progress,
        provider=args.provider
    ))