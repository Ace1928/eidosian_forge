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
def load_layers_(layer_lst: nn.ModuleList, opus_state: dict, converter, is_decoder=False):
    for i, layer in enumerate(layer_lst):
        layer_tag = f'decoder_l{i + 1}_' if is_decoder else f'encoder_l{i + 1}_'
        sd = convert_encoder_layer(opus_state, layer_tag, converter)
        layer.load_state_dict(sd, strict=False)