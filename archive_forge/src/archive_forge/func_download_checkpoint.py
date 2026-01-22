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
def download_checkpoint(path):
    url = 'https://storage.googleapis.com/scenic-bucket/vivit/kinetics_400/vivit_base_16x2_unfactorized/checkpoint'
    with open(path, 'wb') as f:
        with requests.get(url, stream=True) as req:
            for chunk in req.iter_content(chunk_size=2048):
                f.write(chunk)