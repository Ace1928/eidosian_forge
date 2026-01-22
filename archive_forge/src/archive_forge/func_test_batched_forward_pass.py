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
@parameterized.expand([['gpt2'], ['bloom'], ['t5']])
def test_batched_forward_pass(self, name):
    """
        Test if the loss trainer works fine
        """
    dummy_dataset = self._init_dummy_dataset()
    dummy_queries = [torch.tensor([1, 2, 3, 4]), torch.tensor([1, 2, 3, 4, 5, 6, 7])]
    dummy_responses = [torch.tensor([5, 6, 7, 8, 9]), torch.tensor([8, 9, 10, 11, 12, 13])]
    if name == 'gpt2':
        model = self.gpt2_model
        tokenizer = self.gpt2_tokenizer
    elif name == 'bloom':
        model = self.bloom_model
        tokenizer = self.bloom_tokenizer
    elif name == 't5':
        model = self.t5_model
        tokenizer = self.t5_tokenizer
    model.eval()
    ppo_trainer = PPOTrainer(config=self.ppo_config, model=model, ref_model=None, tokenizer=tokenizer, dataset=dummy_dataset)
    ppo_trainer.config.mini_batch_size = 1
    ppo_trainer.config.batch_size = 1
    model_inputs = ppo_trainer.prepare_model_inputs([dummy_queries[0]], [dummy_responses[0]])
    logprobs_0, logits_0, values_0, mask_0 = ppo_trainer.batched_forward_pass(model, [dummy_queries[0]], [dummy_responses[0]], model_inputs)
    ppo_trainer.config.batch_size = 2
    model_inputs = ppo_trainer.prepare_model_inputs(dummy_queries, dummy_responses)
    logprobs_1, logits_1, values_1, mask_1 = ppo_trainer.batched_forward_pass(model, dummy_queries, dummy_responses, model_inputs)
    ppo_trainer.config.mini_batch_size = 2
    model_inputs = ppo_trainer.prepare_model_inputs(dummy_queries, dummy_responses)
    logprobs_2, logits_2, values_2, mask_2 = ppo_trainer.batched_forward_pass(model, dummy_queries, dummy_responses, model_inputs)
    assert abs_diff_masked_tensors(logprobs_1, logprobs_2, mask_1, mask_2) <= 0.0001
    assert abs_diff_masked_tensors(values_1, values_2, mask_1, mask_2) <= 0.0001
    assert abs_diff_masked_tensors(logprobs_0, logprobs_2[:1], mask_0, mask_2[:1]) <= 0.0001
    assert abs_diff_masked_tensors(values_0, values_2[:1], mask_0, mask_2[:1]) <= 0.0001