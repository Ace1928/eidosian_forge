import argparse
import gc
import os
import safetensors
import torch
from huggingface_hub import hf_hub_download
from transformers import PhiConfig, PhiForCausalLM
def convert_weights(original_weights, mapping, config):
    converted_weights = {}
    original_weights_keys = sorted(original_weights.keys())
    for original_weights_key in original_weights_keys:
        new_key = original_weights_key
        if 'rotary_emb' in new_key:
            continue
        if 'Wqkv' in new_key:
            if 'weight' in new_key:
                weight = original_weights[new_key]
                weights_shape = weight.shape
                weight = weight.view(3, config.num_attention_heads, -1, config.hidden_size).transpose(0, 1).reshape(*weights_shape)
                original_weights[new_key] = weight
            elif 'bias' in new_key:
                bias = original_weights[new_key]
                bias_shape = bias.shape
                bias = bias.view(3, config.num_attention_heads, -1).transpose(0, 1).reshape(*bias_shape)
                original_weights[new_key] = bias
        for k, v in mapping.items():
            if k in new_key:
                new_key = new_key.replace(k, v)
        converted_weights[new_key] = original_weights.pop(original_weights_key)
    return converted_weights