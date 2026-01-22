import logging
import re
from collections import OrderedDict
from collections.abc import Sequence
from functools import partial
from typing import Any, Mapping
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers import BertConfig, PretrainedConfig
from transformers.models.bert.modeling_bert import (
from flash_attn.bert_padding import (
from flash_attn.modules.block import Block
from flash_attn.modules.embedding import BertEmbeddings
from flash_attn.modules.mha import MHA
from flash_attn.modules.mlp import FusedMLP, Mlp
from flash_attn.utils.pretrained import state_dict_from_pretrained
def key_mapping_ln_gamma_beta(key):
    key = re.sub('LayerNorm.gamma$', 'LayerNorm.weight', key)
    key = re.sub('LayerNorm.beta$', 'LayerNorm.bias', key)
    return key