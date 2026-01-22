import argparse
import json
import os.path
from collections import OrderedDict
import numpy as np
import requests
import torch
from flax.training.checkpoints import restore_checkpoint
from huggingface_hub import hf_hub_download
from transformers import VivitConfig, VivitForVideoClassification, VivitImageProcessor
from transformers.image_utils import PILImageResampling
def transform_attention(current: np.ndarray):
    if np.ndim(current) == 2:
        return transform_attention_bias(current)
    elif np.ndim(current) == 3:
        return transform_attention_kernel(current)
    else:
        raise Exception(f'Invalid number of dimesions: {np.ndim(current)}')