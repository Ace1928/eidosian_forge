import argparse
import json
import re
from pathlib import Path
import requests
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import (
from transformers.utils import logging
def get_mobilenet_v2_config(model_name):
    config = MobileNetV2Config(layer_norm_eps=0.001)
    if 'quant' in model_name:
        raise ValueError('Quantized models are not supported.')
    matches = re.match('^.*mobilenet_v2_([^_]*)_([^_]*)$', model_name)
    if matches:
        config.depth_multiplier = float(matches[1])
        config.image_size = int(matches[2])
    if model_name.startswith('deeplabv3_'):
        config.output_stride = 8
        config.num_labels = 21
        filename = 'pascal-voc-id2label.json'
    else:
        config.num_labels = 1001
        filename = 'imagenet-1k-id2label.json'
    repo_id = 'huggingface/label-files'
    id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type='dataset'), 'r'))
    if config.num_labels == 1001:
        id2label = {int(k) + 1: v for k, v in id2label.items()}
        id2label[0] = 'background'
    else:
        id2label = {int(k): v for k, v in id2label.items()}
    config.id2label = id2label
    config.label2id = {v: k for k, v in id2label.items()}
    return config