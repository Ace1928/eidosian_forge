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
def test_sft_trainer_uncorrect_data(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
        training_args = TrainingArguments(output_dir=tmp_dir, dataloader_drop_last=True, evaluation_strategy='steps', max_steps=2, eval_steps=1, save_steps=1, per_device_train_batch_size=2)
        with pytest.raises(ValueError):
            _ = SFTTrainer(model=self.model, args=training_args, train_dataset=self.dummy_dataset, packing=True)
        _ = SFTTrainer(model=self.model, args=training_args, train_dataset=self.dummy_chatml_dataset, max_seq_length=32, num_of_sequences=32, packing=True)
        _ = SFTTrainer(model=self.model, args=training_args, train_dataset=self.dummy_chatml_dataset, packing=False)
        _ = SFTTrainer(model=self.model, args=training_args, train_dataset=self.dummy_instruction_dataset, max_seq_length=16, packing=True)
        _ = SFTTrainer(model=self.model, args=training_args, train_dataset=self.dummy_instruction_dataset, packing=False)
        _ = SFTTrainer(model=self.model, args=training_args, train_dataset=self.dummy_dataset, formatting_func=formatting_prompts_func, max_seq_length=32, packing=True)
        with pytest.raises(ValueError):
            _ = SFTTrainer(model=self.model, args=training_args, train_dataset=self.dummy_dataset, formatting_func=formatting_prompts_func, max_seq_length=1024, packing=True)
        with pytest.raises(ValueError):
            _ = SFTTrainer(model=self.model, args=training_args, train_dataset=self.dummy_dataset, formatting_func=formatting_prompts_func, packing=False)
        _ = SFTTrainer(model=self.model, args=training_args, train_dataset=self.dummy_dataset, formatting_func=formatting_prompts_func_batched, packing=False)