import math
import json
import re
from pathlib import Path
from collections import OrderedDict
import torch
import torch.nn.functional as F
from einops import rearrange
from transformers import GPT2Config, AutoConfig, PretrainedConfig
def key_mapping_ln(key):
    key = re.sub('^transformer.ln_f.(weight|bias)', 'transformer.ln_f.\\1', key)
    key = re.sub('^transformer.h.(\\d+).ln_(1|2).(weight|bias)', 'transformer.layers.\\1.norm\\2.\\3', key)
    return key