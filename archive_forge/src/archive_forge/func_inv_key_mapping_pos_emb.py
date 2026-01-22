import math
import re
from collections import OrderedDict
import torch
import torch.nn.functional as F
from transformers import GPT2Config, GPTBigCodeConfig, PretrainedConfig
def inv_key_mapping_pos_emb(key):
    return re.sub('^transformer.embeddings.position_embeddings.', 'transformer.wpe.', key)