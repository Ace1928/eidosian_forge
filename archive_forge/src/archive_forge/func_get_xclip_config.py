import argparse
import gdown
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import (
def get_xclip_config(model_name, num_frames):
    text_config = XCLIPTextConfig()
    start_idx = model_name.find('patch')
    patch_size = int(model_name[start_idx + len('patch'):start_idx + len('patch') + 2])
    vision_config = XCLIPVisionConfig(patch_size=patch_size, num_frames=num_frames)
    if 'large' in model_name:
        text_config.hidden_size = 768
        text_config.intermediate_size = 3072
        text_config.num_attention_heads = 12
        vision_config.hidden_size = 1024
        vision_config.intermediate_size = 4096
        vision_config.num_attention_heads = 16
        vision_config.num_hidden_layers = 24
        vision_config.mit_hidden_size = 768
        vision_config.mit_intermediate_size = 3072
    if model_name == 'xclip-large-patch14-16-frames':
        vision_config.image_size = 336
    config = XCLIPConfig.from_text_vision_configs(text_config, vision_config)
    if 'large' in model_name:
        config.projection_dim = 768
    return config