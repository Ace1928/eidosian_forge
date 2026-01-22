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
def test_ppo_step_input_shape(self):
    """
        Test if the shape of the expected inputs are correct
        """
    dummy_dataset = self._init_dummy_dataset()
    ppo_trainer = PPOTrainer(config=self.ppo_config, model=self.gpt2_model, ref_model=None, tokenizer=self.gpt2_tokenizer, dataset=dummy_dataset)
    dummy_dataloader = ppo_trainer.dataloader
    for query_tensor, response_tensor in dummy_dataloader:
        reward = [torch.tensor([1.0]), torch.tensor([0.0])]
        bs = ppo_trainer.config.batch_size
        queries, responses, _, _ = ppo_trainer._step_safety_checker(bs, list(query_tensor), list(response_tensor), reward)
        assert isinstance(queries, list), f'queries should be a list, got {type(queries)}'
        assert isinstance(responses, list), f'responses should be a list, got {type(responses)}'
        for i in range(bs):
            assert queries[i].shape == torch.Size([7])
            assert responses[i].size() == torch.Size([7])
        break