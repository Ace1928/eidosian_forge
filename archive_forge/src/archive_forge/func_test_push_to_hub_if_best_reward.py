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
@unittest.skip('Fix by either patching `whomai()` to work in the staging endpoint or use a dummy prod user.')
def test_push_to_hub_if_best_reward(self):
    REPO_NAME = 'test-ppo-trainer'
    repo_id = f'{CI_HUB_USER}/{REPO_NAME}'
    dummy_dataset = self._init_dummy_dataset()
    push_to_hub_if_best_kwargs = {'repo_id': repo_id}
    ppo_config = PPOConfig(batch_size=2, mini_batch_size=1, log_with=None, push_to_hub_if_best_kwargs=push_to_hub_if_best_kwargs, compare_steps=1)
    ppo_trainer = PPOTrainer(config=ppo_config, model=self.gpt2_model, ref_model=self.gpt2_model_ref, tokenizer=self.gpt2_tokenizer, dataset=dummy_dataset)
    dummy_dataloader = ppo_trainer.dataloader
    for query_tensor, response_tensor in dummy_dataloader:
        reward = [torch.tensor(1.0), torch.tensor(0.0)]
        _ = ppo_trainer.step(list(query_tensor), list(response_tensor), reward)
        break