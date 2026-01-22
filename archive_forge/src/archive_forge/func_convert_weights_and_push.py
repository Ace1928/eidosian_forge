import argparse
import json
import sys
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import List
import torch
import torch.nn as nn
from huggingface_hub import cached_download, hf_hub_download
from torch import Tensor
from transformers import AutoImageProcessor, VanConfig, VanForImageClassification
from transformers.models.deprecated.van.modeling_van import VanLayerScaling
from transformers.utils import logging
def convert_weights_and_push(save_directory: Path, model_name: str=None, push_to_hub: bool=True):
    filename = 'imagenet-1k-id2label.json'
    num_labels = 1000
    repo_id = 'huggingface/label-files'
    num_labels = num_labels
    id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type='dataset'), 'r'))
    id2label = {int(k): v for k, v in id2label.items()}
    id2label = id2label
    label2id = {v: k for k, v in id2label.items()}
    ImageNetPreTrainedConfig = partial(VanConfig, num_labels=num_labels, id2label=id2label, label2id=label2id)
    names_to_config = {'van-tiny': ImageNetPreTrainedConfig(hidden_sizes=[32, 64, 160, 256], depths=[3, 3, 5, 2], mlp_ratios=[8, 8, 4, 4]), 'van-small': ImageNetPreTrainedConfig(hidden_sizes=[64, 128, 320, 512], depths=[2, 2, 4, 2], mlp_ratios=[8, 8, 4, 4]), 'van-base': ImageNetPreTrainedConfig(hidden_sizes=[64, 128, 320, 512], depths=[3, 3, 12, 3], mlp_ratios=[8, 8, 4, 4]), 'van-large': ImageNetPreTrainedConfig(hidden_sizes=[64, 128, 320, 512], depths=[3, 5, 27, 3], mlp_ratios=[8, 8, 4, 4])}
    names_to_original_models = {'van-tiny': van_tiny, 'van-small': van_small, 'van-base': van_base, 'van-large': van_large}
    names_to_original_checkpoints = {'van-tiny': 'https://huggingface.co/Visual-Attention-Network/VAN-Tiny-original/resolve/main/van_tiny_754.pth.tar', 'van-small': 'https://huggingface.co/Visual-Attention-Network/VAN-Small-original/resolve/main/van_small_811.pth.tar', 'van-base': 'https://huggingface.co/Visual-Attention-Network/VAN-Base-original/resolve/main/van_base_828.pth.tar', 'van-large': 'https://huggingface.co/Visual-Attention-Network/VAN-Large-original/resolve/main/van_large_839.pth.tar'}
    if model_name:
        convert_weight_and_push(model_name, names_to_config[model_name], checkpoint=names_to_original_checkpoints[model_name], from_model=names_to_original_models[model_name](), save_directory=save_directory, push_to_hub=push_to_hub)
    else:
        for model_name, config in names_to_config.items():
            convert_weight_and_push(model_name, config, checkpoint=names_to_original_checkpoints[model_name], from_model=names_to_original_models[model_name](), save_directory=save_directory, push_to_hub=push_to_hub)