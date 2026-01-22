import unittest
from typing import Callable
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl.extras.dataset_formatting import get_formatting_func_from_dataset
from trl.models.utils import ChatMlSpecialTokens, setup_chat_format
def test_example_with_setup_model(self):
    modified_model, modified_tokenizer = setup_chat_format(self.model, self.tokenizer)
    messages = [{'role': 'system', 'content': 'You are helpful'}, {'role': 'user', 'content': 'Hello'}, {'role': 'assistant', 'content': 'Hi, how can I help you?'}]
    prompt = modified_tokenizer.apply_chat_template(messages, tokenize=False)
    assert prompt == '<|im_start|>system\nYou are helpful<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\nHi, how can I help you?<|im_end|>\n'