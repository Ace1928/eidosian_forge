import argparse
import json
from pathlib import Path
import requests
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import ConvNextConfig, ConvNextForImageClassification, ConvNextImageProcessor
from transformers.utils import logging
def get_convnext_config(checkpoint_url):
    config = ConvNextConfig()
    if 'tiny' in checkpoint_url:
        depths = [3, 3, 9, 3]
        hidden_sizes = [96, 192, 384, 768]
    if 'small' in checkpoint_url:
        depths = [3, 3, 27, 3]
        hidden_sizes = [96, 192, 384, 768]
    if 'base' in checkpoint_url:
        depths = [3, 3, 27, 3]
        hidden_sizes = [128, 256, 512, 1024]
    if 'large' in checkpoint_url:
        depths = [3, 3, 27, 3]
        hidden_sizes = [192, 384, 768, 1536]
    if 'xlarge' in checkpoint_url:
        depths = [3, 3, 27, 3]
        hidden_sizes = [256, 512, 1024, 2048]
    if '1k' in checkpoint_url:
        num_labels = 1000
        filename = 'imagenet-1k-id2label.json'
        expected_shape = (1, 1000)
    else:
        num_labels = 21841
        filename = 'imagenet-22k-id2label.json'
        expected_shape = (1, 21841)
    repo_id = 'huggingface/label-files'
    config.num_labels = num_labels
    id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type='dataset'), 'r'))
    id2label = {int(k): v for k, v in id2label.items()}
    if '1k' not in checkpoint_url:
        del id2label[9205]
        del id2label[15027]
    config.id2label = id2label
    config.label2id = {v: k for k, v in id2label.items()}
    config.hidden_sizes = hidden_sizes
    config.depths = depths
    return (config, expected_shape)