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
def find_pretrained_model(src_lang: str, tgt_lang: str) -> List[str]:
    """Find models that can accept src_lang as input and return tgt_lang as output."""
    prefix = 'Helsinki-NLP/opus-mt-'
    model_list = list_models()
    model_ids = [x.modelId for x in model_list if x.modelId.startswith('Helsinki-NLP')]
    src_and_targ = [remove_prefix(m, prefix).lower().split('-') for m in model_ids if '+' not in m]
    matching = [f'{prefix}{a}-{b}' for a, b in src_and_targ if src_lang in a and tgt_lang in b]
    return matching