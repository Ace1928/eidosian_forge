import logging
import math
import re
from collections import OrderedDict, namedtuple
from collections.abc import Sequence
from functools import partial
from typing import Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers import GPT2Config
from flash_attn.models.bigcode import remap_state_dict_hf_bigcode
from flash_attn.models.falcon import remap_state_dict_hf_falcon
from flash_attn.models.gpt_neox import remap_state_dict_hf_gpt_neox
from flash_attn.models.gptj import remap_state_dict_hf_gptj
from flash_attn.models.llama import remap_state_dict_hf_llama
from flash_attn.models.opt import remap_state_dict_hf_opt
from flash_attn.modules.block import Block, ParallelBlock
from flash_attn.modules.embedding import GPT2Embeddings, ParallelGPT2Embeddings
from flash_attn.modules.mha import MHA, ParallelMHA
from flash_attn.modules.mlp import (
from flash_attn.ops.activations import sqrelu_fwd
from flash_attn.utils.distributed import (
from flash_attn.utils.generation import GenerationMixin
from flash_attn.utils.pretrained import state_dict_from_pretrained
def combine_qkv_headdim(state_dicts, state_dict, key):
    n_head = config.n_head
    n_head_kv = getattr(config, 'n_head_kv', n_head)
    if key in state_dict:
        if n_head_kv == n_head:
            xs = [rearrange(s[key], '(three d) ... -> three d ...', three=3) for s in state_dicts]
            state_dict[key] = rearrange(torch.cat(xs, dim=1), 'three d ... -> (three d) ...')
        else:
            n_head_each_rank = [get_dim_for_local_rank(n_head, world_size, local_rank) for local_rank in range(world_size)]
            n_head_kv_each_rank = [get_dim_for_local_rank(n_head_kv, world_size, local_rank) for local_rank in range(world_size)]
            xs = [rearrange(s[key], '(nheadqkv headdim) ... -> nheadqkv headdim ...', nheadqkv=rank_n_head + 2 * rank_n_head_kv, headdim=headdim) for s, rank_n_head, rank_n_head_kv in zip(state_dicts, n_head_each_rank, n_head_kv_each_rank)]
            wq = torch.cat([x[:n_head_each_rank[rank]] for rank, x in enumerate(xs)], dim=0)
            wk = torch.cat([x[n_head_each_rank[rank]:n_head_each_rank[rank] + n_head_kv_each_rank[rank]] for rank, x in enumerate(xs)], dim=0)
            wv = torch.cat([x[n_head_each_rank[rank] + n_head_kv_each_rank[rank]:] for rank, x in enumerate(xs)], dim=0)
            wqkv = torch.cat([wq, wk, wv], dim=0)
            state_dict[key] = rearrange(wqkv, 'nheadqkv headdim ... -> (nheadqkv headdim) ...')