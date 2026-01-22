import argparse
import json
import gdown
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import TimesformerConfig, TimesformerForVideoClassification, VideoMAEImageProcessor
def get_timesformer_config(model_name):
    config = TimesformerConfig()
    if 'large' in model_name:
        config.num_frames = 96
    if 'hr' in model_name:
        config.num_frames = 16
        config.image_size = 448
    repo_id = 'huggingface/label-files'
    if 'k400' in model_name:
        config.num_labels = 400
        filename = 'kinetics400-id2label.json'
    elif 'k600' in model_name:
        config.num_labels = 600
        filename = 'kinetics600-id2label.json'
    elif 'ssv2' in model_name:
        config.num_labels = 174
        filename = 'something-something-v2-id2label.json'
    else:
        raise ValueError("Model name should either contain 'k400', 'k600' or 'ssv2'.")
    id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type='dataset'), 'r'))
    id2label = {int(k): v for k, v in id2label.items()}
    config.id2label = id2label
    config.label2id = {v: k for k, v in id2label.items()}
    return config