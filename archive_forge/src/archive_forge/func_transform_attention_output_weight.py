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
def transform_attention_output_weight(current: np.ndarray):
    return np.reshape(current, (current.shape[0] * current.shape[1], current.shape[2])).T