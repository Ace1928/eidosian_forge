import os
import tempfile
import unittest
import torch
from pytest import mark
from transformers import AutoModelForCausalLM
from trl import AutoModelForCausalLMWithValueHead, is_peft_available
from .testing_utils import require_bitsandbytes, require_peft
def test_create_peft_model_from_config(self):
    """
        Simply creates a peft model and checks that it can be loaded.
        """
    trl_model = AutoModelForCausalLMWithValueHead.from_pretrained(self.causal_lm_model_id, peft_config=self.lora_config)
    nb_trainable_params = sum((p.numel() for p in trl_model.parameters() if p.requires_grad))
    assert nb_trainable_params == 10273
    causal_lm_model = AutoModelForCausalLM.from_pretrained(self.causal_lm_model_id)
    trl_model = AutoModelForCausalLMWithValueHead.from_pretrained(causal_lm_model, peft_config=self.lora_config)
    nb_trainable_params = sum((p.numel() for p in trl_model.parameters() if p.requires_grad))
    assert nb_trainable_params == 10273