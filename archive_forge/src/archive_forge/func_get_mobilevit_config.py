import argparse
import json
from pathlib import Path
import requests
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import (
from transformers.utils import logging
def get_mobilevit_config(mobilevit_name):
    config = MobileViTConfig()
    if 'mobilevit_s' in mobilevit_name:
        config.hidden_sizes = [144, 192, 240]
        config.neck_hidden_sizes = [16, 32, 64, 96, 128, 160, 640]
    elif 'mobilevit_xs' in mobilevit_name:
        config.hidden_sizes = [96, 120, 144]
        config.neck_hidden_sizes = [16, 32, 48, 64, 80, 96, 384]
    elif 'mobilevit_xxs' in mobilevit_name:
        config.hidden_sizes = [64, 80, 96]
        config.neck_hidden_sizes = [16, 16, 24, 48, 64, 80, 320]
        config.hidden_dropout_prob = 0.05
        config.expand_ratio = 2.0
    if mobilevit_name.startswith('deeplabv3_'):
        config.image_size = 512
        config.output_stride = 16
        config.num_labels = 21
        filename = 'pascal-voc-id2label.json'
    else:
        config.num_labels = 1000
        filename = 'imagenet-1k-id2label.json'
    repo_id = 'huggingface/label-files'
    id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type='dataset'), 'r'))
    id2label = {int(k): v for k, v in id2label.items()}
    config.id2label = id2label
    config.label2id = {v: k for k, v in id2label.items()}
    return config