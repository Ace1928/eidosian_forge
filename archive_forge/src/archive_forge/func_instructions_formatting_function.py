import logging
from typing import Callable, Literal, Optional, Union
from datasets import Dataset, Value
from transformers import AutoTokenizer
from ..trainer.utils import ConstantLengthDataset
def instructions_formatting_function(tokenizer: AutoTokenizer):
    """
    return a callable function that takes in an "instructions" dataset and returns a formatted dataset, based on the tokenizer
    apply chat template to the dataset
    """

    def format_dataset(examples):
        if isinstance(examples['prompt'], list):
            output_texts = []
            for i in range(len(examples['prompt'])):
                converted_sample = [{'role': 'user', 'content': examples['prompt'][i]}, {'role': 'assistant', 'content': examples['completion'][i]}]
                output_texts.append(tokenizer.apply_chat_template(converted_sample, tokenize=False))
            return output_texts
        else:
            converted_sample = [{'role': 'user', 'content': examples['prompt']}, {'role': 'assistant', 'content': examples['completion']}]
            return tokenizer.apply_chat_template(converted_sample, tokenize=False)
    return format_dataset