import argparse
import json
import requests
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import ViTImageProcessor, ViTMSNConfig, ViTMSNModel
from transformers.image_utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
def remove_projection_head(state_dict):
    ignore_keys = ['module.fc.fc1.weight', 'module.fc.fc1.bias', 'module.fc.bn1.weight', 'module.fc.bn1.bias', 'module.fc.bn1.running_mean', 'module.fc.bn1.running_var', 'module.fc.bn1.num_batches_tracked', 'module.fc.fc2.weight', 'module.fc.fc2.bias', 'module.fc.bn2.weight', 'module.fc.bn2.bias', 'module.fc.bn2.running_mean', 'module.fc.bn2.running_var', 'module.fc.bn2.num_batches_tracked', 'module.fc.fc3.weight', 'module.fc.fc3.bias']
    for k in ignore_keys:
        state_dict.pop(k, None)