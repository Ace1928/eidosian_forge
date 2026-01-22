import tempfile
import unittest
import torch
from datasets import Dataset
from parameterized import parameterized
from pytest import mark
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, TrainingArguments
from trl import DPOTrainer
from .testing_utils import require_bitsandbytes, require_no_wandb, require_peft
@parameterized.expand([['gpt2', 'sigmoid', False, False], ['gpt2', 'sigmoid', False, True], ['gpt2', 'sigmoid', True, False], ['gpt2', 'sigmoid', True, True], ['gpt2', 'ipo', False, False], ['gpt2', 'ipo', False, True], ['gpt2', 'ipo', True, False], ['gpt2', 'ipo', True, True], ['gpt2', 'kto_pair', False, False], ['gpt2', 'kto_pair', False, True], ['gpt2', 'kto_pair', True, False], ['gpt2', 'kto_pair', True, True]])
@require_bitsandbytes
@require_peft
@mark.peft_test
@unittest.skip('You need a GPU with bf16 support in order to run these tests')
def test_dpo_lora_bf16_autocast(self, name, loss_type, pre_compute, gen_during_eval):
    from peft import LoraConfig
    lora_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias='none', task_type='CAUSAL_LM')
    model = AutoModelForCausalLM.from_pretrained(self.model_id, load_in_4bit=True)
    with tempfile.TemporaryDirectory() as tmp_dir:
        training_args = TrainingArguments(output_dir=tmp_dir, per_device_train_batch_size=2, max_steps=3, remove_unused_columns=False, gradient_accumulation_steps=4, learning_rate=0.9, evaluation_strategy='steps', bf16=True)
        dummy_dataset = self._init_dummy_dataset()
        trainer = DPOTrainer(model=model, ref_model=None, beta=0.1, args=training_args, tokenizer=self.tokenizer, train_dataset=dummy_dataset, eval_dataset=dummy_dataset, peft_config=lora_config, generate_during_eval=gen_during_eval, loss_type=loss_type, precompute_ref_log_probs=pre_compute)
        trainer.train()
        trainer.save_model()