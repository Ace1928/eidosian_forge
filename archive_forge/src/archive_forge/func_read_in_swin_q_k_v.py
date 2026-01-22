import argparse
import json
from pathlib import Path
import requests
import torch
from huggingface_hub import cached_download, hf_hub_download, hf_hub_url
from PIL import Image
from transformers import DetaConfig, DetaForObjectDetection, DetaImageProcessor, SwinConfig
from transformers.utils import logging
def read_in_swin_q_k_v(state_dict, backbone_config):
    num_features = [int(backbone_config.embed_dim * 2 ** i) for i in range(len(backbone_config.depths))]
    for i in range(len(backbone_config.depths)):
        dim = num_features[i]
        for j in range(backbone_config.depths[i]):
            in_proj_weight = state_dict.pop(f'backbone.0.body.layers.{i}.blocks.{j}.attn.qkv.weight')
            in_proj_bias = state_dict.pop(f'backbone.0.body.layers.{i}.blocks.{j}.attn.qkv.bias')
            state_dict[f'model.backbone.model.encoder.layers.{i}.blocks.{j}.attention.self.query.weight'] = in_proj_weight[:dim, :]
            state_dict[f'model.backbone.model.encoder.layers.{i}.blocks.{j}.attention.self.query.bias'] = in_proj_bias[:dim]
            state_dict[f'model.backbone.model.encoder.layers.{i}.blocks.{j}.attention.self.key.weight'] = in_proj_weight[dim:dim * 2, :]
            state_dict[f'model.backbone.model.encoder.layers.{i}.blocks.{j}.attention.self.key.bias'] = in_proj_bias[dim:dim * 2]
            state_dict[f'model.backbone.model.encoder.layers.{i}.blocks.{j}.attention.self.value.weight'] = in_proj_weight[-dim:, :]
            state_dict[f'model.backbone.model.encoder.layers.{i}.blocks.{j}.attention.self.value.bias'] = in_proj_bias[-dim:]