import copy
import os
import tempfile
import unittest
import numpy as np
import pytest
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer
from trl.import_utils import is_peft_available
from trl.trainer import ConstantLengthDataset, DataCollatorForCompletionOnlyLM
from .testing_utils import require_peft
def test_constant_length_dataset(self):
    formatted_dataset = ConstantLengthDataset(self.tokenizer, self.dummy_dataset, dataset_text_field=None, formatting_func=formatting_prompts_func)
    assert len(formatted_dataset) == len(self.dummy_dataset)
    assert len(formatted_dataset) > 0
    for example in formatted_dataset:
        assert 'input_ids' in example
        assert 'labels' in example
        assert len(example['input_ids']) == formatted_dataset.seq_length
        assert len(example['labels']) == formatted_dataset.seq_length
        decoded_text = self.tokenizer.decode(example['input_ids'])
        assert 'Question' in decoded_text and 'Answer' in decoded_text