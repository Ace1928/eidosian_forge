import argparse
import os
import torch
from huggingface_hub import hf_hub_download
from transformers import ClvpConfig, ClvpModelForConditionalGeneration
def convert_encoder_weights(original_weights):
    converted_weights = {}
    original_weights_keys = sorted(original_weights.keys())
    for original_key in original_weights_keys:
        updated_key = original_key
        if '0.0.g' in updated_key:
            present_index = updated_key.split('.')[4]
            if int(present_index) % 2 == 0:
                updated_key = updated_key.replace('0.0.g', 'input_rmsnorm.weight')
            else:
                updated_key = updated_key.replace('0.0.g', 'post_attention_rmsnorm.weight')
        if 'transformer.attn_layers.layers' in updated_key:
            present_index = updated_key.split('.')[4]
            updated_index = update_index(int(present_index))
            updated_key = updated_key.replace(f'transformer.attn_layers.layers.{present_index}', f'transformer.attn_layers.layers.{updated_index}')
        for k, v in CLVP_ENCODERS_MAPPING.items():
            if k in updated_key:
                updated_key = updated_key.replace(k, v)
        converted_weights[updated_key] = original_weights.pop(original_key)
    return converted_weights