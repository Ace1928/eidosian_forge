import unittest
from typing import Callable
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl.extras.dataset_formatting import get_formatting_func_from_dataset
from trl.models.utils import ChatMlSpecialTokens, setup_chat_format
def test_get_formatting_func_from_dataset_with_unknown_format(self):
    dataset = Dataset.from_dict({'text': 'test'})
    formatting_func = get_formatting_func_from_dataset(dataset, self.llama_tokenizer)
    assert formatting_func is None