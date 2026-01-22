import tempfile
import unittest
import torch
from datasets import Dataset
from parameterized import parameterized
from pytest import mark
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, TrainingArguments
from trl import DPOTrainer
from .testing_utils import require_bitsandbytes, require_no_wandb, require_peft
def test_dpo_trainer_w_dataset_num_proc(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
        training_args = TrainingArguments(output_dir=tmp_dir, per_device_train_batch_size=2, max_steps=3, remove_unused_columns=False, gradient_accumulation_steps=1, learning_rate=0.9, evaluation_strategy='steps')
        dummy_dataset = self._init_dummy_dataset()
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        tokenizer.pad_token = None
        with self.assertRaisesRegex(ValueError, expected_regex='Padding is enabled, but the tokenizer is not configured with a padding token. Explicitly set `tokenizer.pad_token` \\(e.g. `tokenizer.pad_token = tokenizer.eos_token`\\) before calling the trainer.'):
            trainer = DPOTrainer(model=self.model, ref_model=None, beta=0.1, args=training_args, tokenizer=tokenizer, train_dataset=dummy_dataset, eval_dataset=dummy_dataset, dataset_num_proc=5)
            trainer.train()