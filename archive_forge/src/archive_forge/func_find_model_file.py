import argparse
import json
import os
import socket
import time
import warnings
from pathlib import Path
from typing import Dict, List, Union
from zipfile import ZipFile
import numpy as np
import torch
from huggingface_hub.hf_api import list_models
from torch import nn
from tqdm import tqdm
from transformers import MarianConfig, MarianMTModel, MarianTokenizer
def find_model_file(dest_dir):
    model_files = list(Path(dest_dir).glob('*.npz'))
    if len(model_files) != 1:
        raise ValueError(f'Found more than one model file: {model_files}')
    model_file = model_files[0]
    return model_file