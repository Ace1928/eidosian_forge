import fire
import os
import sys
import time
import torch
from transformers import AutoTokenizer
from llama_recipes.inference.safety_utils import get_safety_checker
from llama_recipes.inference.model_utils import load_model, load_peft_model
def print_safety_results(are_safe_prompt, prompt, safety_results, prompt_type='User'):
    """
        Prints the safety results for a prompt.

        Parameters:
        - are_safe_prompt (bool): Indicates whether the prompt is safe.
        - prompt (str): The prompt that was checked for safety.
        - safety_results (list of tuples): A list of tuples containing the method, safety status, and safety report.
        - prompt_type (str): The type of prompt (User/System).
        """
    if are_safe_prompt:
        print(f'{prompt_type} prompt deemed safe.')
        print(f'{prompt_type} prompt:\n{prompt}')
    else:
        print(f'{prompt_type} prompt deemed unsafe.')
        for method, is_safe, report in safety_results:
            if not is_safe:
                print(method)
                print(report)
        print(f'Skipping the inference as the {prompt_type.lower()} prompt is not safe.')
        sys.exit(1)