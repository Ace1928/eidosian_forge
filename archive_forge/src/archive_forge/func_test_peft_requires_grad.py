import os
import tempfile
import unittest
import torch
from pytest import mark
from transformers import AutoModelForCausalLM
from trl import AutoModelForCausalLMWithValueHead, is_peft_available
from .testing_utils import require_bitsandbytes, require_peft
def test_peft_requires_grad(self):
    """
        Check that the value head of the returned model has requires_grad=True.
        """
    causal_lm_model = AutoModelForCausalLM.from_pretrained(self.causal_lm_model_id)
    pretrained_model = get_peft_model(causal_lm_model, self.lora_config)
    model = AutoModelForCausalLMWithValueHead.from_pretrained(pretrained_model)
    assert model.v_head.summary.weight.requires_grad