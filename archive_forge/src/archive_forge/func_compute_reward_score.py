import json
import logging
import os
from copy import deepcopy
from typing import Optional
import torch
import torch.nn as nn
from accelerate import PartialState
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import (
from safetensors.torch import load_file as safe_load_file
from transformers import PreTrainedModel
from ..import_utils import is_npu_available, is_peft_available, is_transformers_greater_than, is_xpu_available
def compute_reward_score(self, input_ids, attention_mask=None, **kwargs):
    """
        Computes the reward score for a given input. The method has first to enable the adapter
        and then compute the reward score. After that the model disables the reward modeling
        adapter and enables the default ppo adapter again.
        """
    if not self.supports_rm_adapter:
        raise ValueError('This model does not support reward modeling adapter.')
    self.pretrained_model.set_adapter(self.rm_adapter_name)
    self.pretrained_model.eval()
    with torch.no_grad():
        base_model_output = self.pretrained_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, return_dict=True, **kwargs)
        last_hidden_states = base_model_output.hidden_states[-1]
        scores = self.score(last_hidden_states)
    self.pretrained_model.set_adapter(self.policy_adapter_name)
    self.pretrained_model.eval()
    return scores