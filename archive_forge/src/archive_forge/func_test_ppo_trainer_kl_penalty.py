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
def test_ppo_trainer_kl_penalty(self):
    dummy_dataset = self._init_dummy_dataset()
    log_probs = torch.Tensor([[0.5, 0.2, 0.1], [0.6, 0.2, 0.1]])
    ref_log_probs = torch.Tensor([[0.4, 0.3, 0.0], [0.7, 0.1, 0.3]])
    ppo_trainer = PPOTrainer(config=self.ppo_config, model=self.gpt2_model, ref_model=None, tokenizer=self.gpt2_tokenizer, dataset=dummy_dataset)
    expected_output = torch.Tensor([[0.1, -0.1, 0.1], [-0.1, 0.1, -0.2]])
    assert torch.allclose(ppo_trainer._kl_penalty(log_probs, ref_log_probs), expected_output)
    self.ppo_config.kl_penalty = 'abs'
    ppo_trainer = PPOTrainer(config=self.ppo_config, model=self.gpt2_model, ref_model=None, tokenizer=self.gpt2_tokenizer, dataset=dummy_dataset)
    expected_output = torch.Tensor([[0.1, 0.1, 0.1], [0.1, 0.1, 0.2]])
    assert torch.allclose(ppo_trainer._kl_penalty(log_probs, ref_log_probs), expected_output)
    self.ppo_config.kl_penalty = 'mse'
    ppo_trainer = PPOTrainer(config=self.ppo_config, model=self.gpt2_model, ref_model=None, tokenizer=self.gpt2_tokenizer, dataset=dummy_dataset)
    expected_output = torch.Tensor([[0.005, 0.005, 0.005], [0.005, 0.005, 0.02]])
    assert torch.allclose(ppo_trainer._kl_penalty(log_probs, ref_log_probs), expected_output)