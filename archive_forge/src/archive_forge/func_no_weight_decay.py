import math
import re
from collections import OrderedDict
from copy import deepcopy
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.helpers import named_apply
from torch.nn.init import trunc_normal_
from torchvision.ops import StochasticDepth
from flash_attn.layers.patch_embed import PatchEmbed
from flash_attn.modules.block import Block
from flash_attn.modules.mha import MHA
from flash_attn.modules.mlp import FusedMLP, Mlp
@torch.jit.ignore
def no_weight_decay(self):
    return {'pos_embed', 'cls_token'}