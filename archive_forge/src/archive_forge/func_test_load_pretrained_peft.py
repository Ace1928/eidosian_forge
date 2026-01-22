import os
import tempfile
import unittest
import torch
from pytest import mark
from transformers import AutoModelForCausalLM
from trl import AutoModelForCausalLMWithValueHead, is_peft_available
from .testing_utils import require_bitsandbytes, require_peft
def test_load_pretrained_peft(self):
    """
        Check that the model saved with peft class interface can be loaded properly.
        """
    causal_lm_model = AutoModelForCausalLM.from_pretrained(self.causal_lm_model_id)
    pretrained_model = get_peft_model(causal_lm_model, self.lora_config)
    model = AutoModelForCausalLMWithValueHead.from_pretrained(pretrained_model)
    with tempfile.TemporaryDirectory() as tmp_dir:
        pretrained_model.save_pretrained(tmp_dir)
        model_from_pretrained = AutoModelForCausalLMWithValueHead.from_pretrained(tmp_dir)
        assert os.path.isfile(f'{tmp_dir}/adapter_model.safetensors'), f'{tmp_dir}/adapter_model.safetensors does not exist'
        assert os.path.exists(f'{tmp_dir}/adapter_config.json'), f'{tmp_dir}/adapter_config.json does not exist'
        for p1, p2 in zip(model.named_parameters(), model_from_pretrained.named_parameters()):
            if p1[0] not in ['v_head.summary.weight', 'v_head.summary.bias']:
                assert torch.allclose(p1[1], p2[1]), f'{p1[0]} != {p2[0]}'