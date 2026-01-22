import argparse
import json
from pathlib import Path
import requests
import torch
from huggingface_hub import cached_download, hf_hub_download, hf_hub_url
from PIL import Image
from transformers import DetaConfig, DetaForObjectDetection, DetaImageProcessor
from transformers.utils import logging
def read_in_decoder_q_k_v(state_dict, config):
    hidden_size = config.d_model
    for i in range(config.decoder_layers):
        in_proj_weight = state_dict.pop(f'transformer.decoder.layers.{i}.self_attn.in_proj_weight')
        in_proj_bias = state_dict.pop(f'transformer.decoder.layers.{i}.self_attn.in_proj_bias')
        state_dict[f'model.decoder.layers.{i}.self_attn.q_proj.weight'] = in_proj_weight[:hidden_size, :]
        state_dict[f'model.decoder.layers.{i}.self_attn.q_proj.bias'] = in_proj_bias[:hidden_size]
        state_dict[f'model.decoder.layers.{i}.self_attn.k_proj.weight'] = in_proj_weight[hidden_size:hidden_size * 2, :]
        state_dict[f'model.decoder.layers.{i}.self_attn.k_proj.bias'] = in_proj_bias[hidden_size:hidden_size * 2]
        state_dict[f'model.decoder.layers.{i}.self_attn.v_proj.weight'] = in_proj_weight[-hidden_size:, :]
        state_dict[f'model.decoder.layers.{i}.self_attn.v_proj.bias'] = in_proj_bias[-hidden_size:]