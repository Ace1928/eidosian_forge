import tempfile
import unittest
import torch
from datasets import Dataset
from parameterized import parameterized
from pytest import mark
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, TrainingArguments
from trl import DPOTrainer
from .testing_utils import require_bitsandbytes, require_no_wandb, require_peft
@require_peft
@mark.peft_test
def test_dpo_lora_save(self):
    from peft import LoraConfig, get_peft_model
    lora_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias='none', task_type='CAUSAL_LM')
    model = AutoModelForCausalLM.from_pretrained(self.model_id)
    model_peft = get_peft_model(model, lora_config)
    with tempfile.TemporaryDirectory() as tmp_dir:
        training_args = TrainingArguments(output_dir=tmp_dir, per_device_train_batch_size=2, max_steps=3, remove_unused_columns=False, gradient_accumulation_steps=4, learning_rate=0.9, evaluation_strategy='steps')
        dummy_dataset = self._init_dummy_dataset()
        trainer = DPOTrainer(model=model_peft, ref_model=None, beta=0.1, args=training_args, tokenizer=self.tokenizer, train_dataset=dummy_dataset, eval_dataset=dummy_dataset, peft_config=lora_config, precompute_ref_log_probs=True)
        trainer.train()
        trainer.save_model()
        try:
            AutoModelForCausalLM.from_pretrained(tmp_dir)
        except OSError:
            self.fail('Loading the saved peft adapter failed')