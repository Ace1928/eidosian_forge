import argparse
from pathlib import Path
import requests
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import DepthAnythingConfig, DepthAnythingForDepthEstimation, Dinov2Config, DPTImageProcessor
from transformers.utils import logging
def get_dpt_config(model_name):
    if 'small' in model_name:
        backbone_config = Dinov2Config.from_pretrained('facebook/dinov2-small', out_indices=[9, 10, 11, 12], apply_layernorm=True, reshape_hidden_states=False)
        fusion_hidden_size = 64
        neck_hidden_sizes = [48, 96, 192, 384]
    elif 'base' in model_name:
        backbone_config = Dinov2Config.from_pretrained('facebook/dinov2-base', out_indices=[9, 10, 11, 12], apply_layernorm=True, reshape_hidden_states=False)
        fusion_hidden_size = 128
        neck_hidden_sizes = [96, 192, 384, 768]
    elif 'large' in model_name:
        backbone_config = Dinov2Config.from_pretrained('facebook/dinov2-large', out_indices=[21, 22, 23, 24], apply_layernorm=True, reshape_hidden_states=False)
        fusion_hidden_size = 256
        neck_hidden_sizes = [256, 512, 1024, 1024]
    else:
        raise NotImplementedError('To do')
    config = DepthAnythingConfig(reassemble_hidden_size=backbone_config.hidden_size, patch_size=backbone_config.patch_size, backbone_config=backbone_config, fusion_hidden_size=fusion_hidden_size, neck_hidden_sizes=neck_hidden_sizes)
    return config