import os
import tempfile
import unittest
import torch
from pytest import mark
from transformers import AutoModelForCausalLM
from trl import AutoModelForCausalLMWithValueHead, is_peft_available
from .testing_utils import require_bitsandbytes, require_peft
@require_bitsandbytes
def test_create_bnb_peft_model_from_config(self):
    """
        Simply creates a peft model and checks that it can be loaded.
        """
    from bitsandbytes.nn import Linear8bitLt
    trl_model = AutoModelForCausalLMWithValueHead.from_pretrained(self.causal_lm_model_id, peft_config=self.lora_config, load_in_8bit=True)
    nb_trainable_params = sum((p.numel() for p in trl_model.parameters() if p.requires_grad))
    assert nb_trainable_params == 10273
    assert trl_model.pretrained_model.model.gpt_neox.layers[0].mlp.dense_h_to_4h.__class__ == Linear8bitLt
    causal_lm_model = AutoModelForCausalLM.from_pretrained(self.causal_lm_model_id, load_in_8bit=True, device_map='auto')
    trl_model = AutoModelForCausalLMWithValueHead.from_pretrained(causal_lm_model, peft_config=self.lora_config)
    nb_trainable_params = sum((p.numel() for p in trl_model.parameters() if p.requires_grad))
    assert nb_trainable_params == 10273
    assert trl_model.pretrained_model.model.gpt_neox.layers[0].mlp.dense_h_to_4h.__class__ == Linear8bitLt