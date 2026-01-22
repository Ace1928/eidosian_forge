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
def formatting_prompts_func_batched(example):
    output_text = []
    for i, question in enumerate(example['question']):
        text = f'### Question: {question}\n ### Answer: {example['answer'][i]}'
        output_text.append(text)
    return output_text