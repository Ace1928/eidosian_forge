import logging
from typing import Callable, Literal, Optional, Union
from datasets import Dataset, Value
from transformers import AutoTokenizer
from ..trainer.utils import ConstantLengthDataset
def conversations_formatting_function(tokenizer: AutoTokenizer, messages_field: Literal['messages', 'conversations']):
    """
    return a callable function that takes in a "messages" dataset and returns a formatted dataset, based on the tokenizer
    apply chat template to the dataset
    """

    def format_dataset(examples):
        if isinstance(examples[messages_field][0], list):
            output_texts = []
            for i in range(len(examples[messages_field])):
                output_texts.append(tokenizer.apply_chat_template(examples[messages_field][i], tokenize=False))
            return output_texts
        else:
            return tokenizer.apply_chat_template(examples[messages_field], tokenize=False)
    return format_dataset