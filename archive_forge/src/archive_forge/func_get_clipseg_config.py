import argparse
import requests
import torch
from PIL import Image
from transformers import (
def get_clipseg_config(model_name):
    text_config = CLIPSegTextConfig()
    vision_config = CLIPSegVisionConfig(patch_size=16)
    use_complex_transposed_convolution = True if 'refined' in model_name else False
    reduce_dim = 16 if 'rd16' in model_name else 64
    config = CLIPSegConfig.from_text_vision_configs(text_config, vision_config, use_complex_transposed_convolution=use_complex_transposed_convolution, reduce_dim=reduce_dim)
    return config