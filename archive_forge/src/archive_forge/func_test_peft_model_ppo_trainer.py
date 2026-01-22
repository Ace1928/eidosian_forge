import copy
import fnmatch
import gc
import re
import tempfile
import unittest
import pytest
import torch
from huggingface_hub import HfApi, HfFolder, delete_repo
from parameterized import parameterized
from pytest import mark
from requests.exceptions import HTTPError
from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead, AutoModelForSeq2SeqLMWithValueHead, PPOConfig, PPOTrainer, set_seed
from trl.core import respond_to_batch
from .testing_constants import CI_HUB_ENDPOINT, CI_HUB_USER, CI_HUB_USER_TOKEN
from .testing_utils import require_peft, require_torch_multi_gpu
@require_peft
@mark.peft_test
def test_peft_model_ppo_trainer(self):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM
    lora_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias='none', task_type='CAUSAL_LM')
    gpt2_model = AutoModelForCausalLM.from_pretrained(self.model_id)

    def make_inputs_require_grad(module, input, output):
        output.requires_grad_(True)
    gpt2_model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    peft_model = get_peft_model(gpt2_model, lora_config)
    model = AutoModelForCausalLMWithValueHead.from_pretrained(peft_model)
    dummy_dataset = self._init_dummy_dataset()
    self.ppo_config.batch_size = 2
    self.ppo_config.mini_batch_size = 1
    ppo_trainer = PPOTrainer(config=self.ppo_config, model=model, ref_model=None, tokenizer=self.gpt2_tokenizer, dataset=dummy_dataset)
    assert ppo_trainer.ref_model is None
    dummy_dataloader = ppo_trainer.dataloader
    for query_tensor, response_tensor in dummy_dataloader:
        reward = [torch.tensor(1.0), torch.tensor(0.0)]
        _ = ppo_trainer.step(list(query_tensor), list(response_tensor), reward)
        ppo_trainer.model.train()
        ppo_trainer.model.gradient_checkpointing_enable()
        _ = ppo_trainer.step(list(query_tensor), list(response_tensor), reward)
        break
    for name, param in model.named_parameters():
        if 'lora' in name or 'v_head' in name:
            assert param.grad is not None, f'Parameter {name} has a no gradient'
        else:
            assert param.grad is None, f'Parameter {name} has a gradient'