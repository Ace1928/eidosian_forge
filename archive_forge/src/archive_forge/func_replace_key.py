import argparse
import json
import os
from pathlib import Path
import requests
import torch
from transformers import JukeboxConfig, JukeboxModel
from transformers.utils import logging
def replace_key(key):
    if key.endswith('.model.1.bias') and len(key.split('.')) > 10:
        key = key.replace('.model.1.bias', '.conv1d_1.bias')
    elif key.endswith('.model.1.weight') and len(key.split('.')) > 10:
        key = key.replace('.model.1.weight', '.conv1d_1.weight')
    elif key.endswith('.model.3.bias') and len(key.split('.')) > 10:
        key = key.replace('.model.3.bias', '.conv1d_2.bias')
    elif key.endswith('.model.3.weight') and len(key.split('.')) > 10:
        key = key.replace('.model.3.weight', '.conv1d_2.weight')
    if 'conditioner_blocks.0.' in key:
        key = key.replace('conditioner_blocks.0', 'conditioner_blocks')
    if 'prime_prior' in key:
        key = key.replace('prime_prior', 'encoder')
    if '.emb.' in key and 'total' not in key and ('absolute' not in key) and ('relative' not in key):
        key = key.replace('.emb.', '.')
    if key.endswith('k'):
        return key.replace('.k', '.codebook')
    if 'y_emb.' in key:
        return key.replace('y_emb.', 'metadata_embedding.')
    if 'x_emb.emb.' in key:
        key = key.replace('0.x_emb.emb', 'embed_tokens')
    if 'prime_state_ln' in key:
        return key.replace('prime_state_ln', 'encoder.final_layer_norm')
    if '.ln' in key:
        return key.replace('.ln', '.layer_norm')
    if '_ln' in key:
        return key.replace('_ln', '_layer_norm')
    if 'prime_state_proj' in key:
        return key.replace('prime_state_proj', 'encoder.proj_in')
    if 'prime_x_out' in key:
        return key.replace('prime_x_out', 'encoder.lm_head')
    if 'prior.x_out' in key:
        return key.replace('x_out', 'fc_proj_out')
    if 'x_emb' in key:
        return key.replace('x_emb', 'embed_tokens')
    return key