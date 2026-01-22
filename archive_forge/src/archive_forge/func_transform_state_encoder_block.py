import argparse
import json
import os.path
from collections import OrderedDict
import numpy as np
import requests
import torch
from flax.training.checkpoints import restore_checkpoint
from huggingface_hub import hf_hub_download
from transformers import VivitConfig, VivitForVideoClassification, VivitImageProcessor
from transformers.image_utils import PILImageResampling
def transform_state_encoder_block(state_dict, i):
    state = state_dict['optimizer']['target']['Transformer'][f'encoderblock_{i}']
    prefix = f'encoder.layer.{i}.'
    new_state = {prefix + 'intermediate.dense.bias': state['MlpBlock_0']['Dense_0']['bias'], prefix + 'intermediate.dense.weight': np.transpose(state['MlpBlock_0']['Dense_0']['kernel']), prefix + 'output.dense.bias': state['MlpBlock_0']['Dense_1']['bias'], prefix + 'output.dense.weight': np.transpose(state['MlpBlock_0']['Dense_1']['kernel']), prefix + 'layernorm_before.bias': state['LayerNorm_0']['bias'], prefix + 'layernorm_before.weight': state['LayerNorm_0']['scale'], prefix + 'layernorm_after.bias': state['LayerNorm_1']['bias'], prefix + 'layernorm_after.weight': state['LayerNorm_1']['scale'], prefix + 'attention.attention.query.bias': transform_attention(state['MultiHeadDotProductAttention_0']['query']['bias']), prefix + 'attention.attention.query.weight': transform_attention(state['MultiHeadDotProductAttention_0']['query']['kernel']), prefix + 'attention.attention.key.bias': transform_attention(state['MultiHeadDotProductAttention_0']['key']['bias']), prefix + 'attention.attention.key.weight': transform_attention(state['MultiHeadDotProductAttention_0']['key']['kernel']), prefix + 'attention.attention.value.bias': transform_attention(state['MultiHeadDotProductAttention_0']['value']['bias']), prefix + 'attention.attention.value.weight': transform_attention(state['MultiHeadDotProductAttention_0']['value']['kernel']), prefix + 'attention.output.dense.bias': state['MultiHeadDotProductAttention_0']['out']['bias'], prefix + 'attention.output.dense.weight': transform_attention_output_weight(state['MultiHeadDotProductAttention_0']['out']['kernel'])}
    return new_state