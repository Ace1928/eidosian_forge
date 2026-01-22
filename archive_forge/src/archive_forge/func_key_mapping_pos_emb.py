import math
import json
import re
from pathlib import Path
from collections import OrderedDict
import torch
import torch.nn.functional as F
from einops import rearrange
from transformers import GPT2Config, AutoConfig, PretrainedConfig
def key_mapping_pos_emb(key):
    return re.sub('^transformer.wpe.', 'transformer.embeddings.position_embeddings.', key)