import math
import json
import re
from pathlib import Path
from collections import OrderedDict
import torch
import torch.nn.functional as F
from einops import rearrange
from transformers import GPT2Config, AutoConfig, PretrainedConfig
def key_mapping_attn(key):
    key = re.sub('^transformer.h.(\\d+).attn.c_attn.bias', 'transformer.layers.\\1.mixer.Wqkv.bias', key)
    key = re.sub('^transformer.h.(\\d+).attn.c_proj.bias', 'transformer.layers.\\1.mixer.out_proj.bias', key)
    return key