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
def test_ppo_step_with_ref_and_custom_layers_warning(self):
    """
        Test PPO step with a reference model and custom layers
        The trainer should raise a warning if the argument `num_shared_layers` is set
        together with a reference model.
        """
    dummy_dataset = self._init_dummy_dataset()
    num_shared_layers = 6
    with self.assertWarns(UserWarning):
        _ = PPOTrainer(config=self.ppo_config, model=self.gpt2_model, ref_model=self.gpt2_model_ref, tokenizer=self.gpt2_tokenizer, dataset=dummy_dataset, num_shared_layers=num_shared_layers)