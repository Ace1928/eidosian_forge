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
def test_ppo_step_no_dataset(self):
    """
        Test if the training loop works fine without passing a dataset
        """
    query_txt = 'This morning I went to the '
    query_tensor = self.gpt2_tokenizer.encode(query_txt, return_tensors='pt')
    self.ppo_config.batch_size = 1
    response_tensor = respond_to_batch(self.gpt2_model, query_tensor)
    with self.assertWarns(UserWarning):
        ppo_trainer = PPOTrainer(config=self.ppo_config, model=self.gpt2_model, ref_model=self.gpt2_model_ref, tokenizer=self.gpt2_tokenizer)
    reward = [torch.tensor([1.0])]
    train_stats = ppo_trainer.step([query_tensor[0]], [response_tensor[0]], reward)
    for name, param in ppo_trainer.model.named_parameters():
        assert param.grad is not None, f'Parameter {name} has no gradient'
    for name, param in ppo_trainer.ref_model.named_parameters():
        assert param.grad is None, f'Parameter {name} has a gradient'
    for stat in EXPECTED_STATS:
        assert stat in train_stats, f'Train stats should contain {stat}'