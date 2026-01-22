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
def test_data_collator_completion_lm_with_multiple_text(self):
    tokenizer = copy.deepcopy(self.tokenizer)
    tokenizer.padding_side = 'left'
    response_template = '### Response:\n'
    data_collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer, mlm=False)
    text1 = '\n\n### Instructions:\nHello all this should be masked\n\n### Response:\nI have not been masked correctly.'
    text2 = '\n\n### Instructions:\nThis is another longer text that should also be masked. This text is significantly longer than the previous one.\n\n### Response:\nI have not been masked correctly.'
    encoded_text1 = tokenizer(text1)
    encoded_text2 = tokenizer(text2)
    examples = [encoded_text1, encoded_text2]
    batch = data_collator(examples)
    for i in range(2):
        labels = batch['labels'][i]
        last_pad_idx = np.where(labels == -100)[0][-1]
        result_text = tokenizer.decode(batch['input_ids'][i, last_pad_idx + 1:])
        assert result_text == 'I have not been masked correctly.'