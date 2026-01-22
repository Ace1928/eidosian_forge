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
def inv_key_mapping_mlp(key):
    key = re.sub('bert.encoder.layer.(\\d+).mlp.fc1.(weight|bias)', 'bert.encoder.layer.\\1.intermediate.dense.\\2', key)
    key = re.sub('bert.encoder.layer.(\\d+).mlp.fc2.(weight|bias)', 'bert.encoder.layer.\\1.output.dense.\\2', key)
    return key