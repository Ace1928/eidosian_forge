import math
import json
import re
from pathlib import Path
from collections import OrderedDict
import torch
import torch.nn.functional as F
from einops import rearrange
from transformers import GPT2Config, AutoConfig, PretrainedConfig
def key_mapping_mlp(key):
    key = re.sub('^transformer.h.(\\d+).mlp.c_proj.bias', 'transformer.layers.\\1.mlp.fc2.bias', key)
    return key