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
def create_mixer_cls(config, layer_idx=None, process_group=None, device=None, dtype=None):
    factory_kwargs = {'device': device, 'dtype': dtype}
    head_dim = getattr(config, 'head_dim', config.hidden_size // config.num_attention_heads)
    attn_scale_power = 0.5 if not getattr(config, 'mup_scale_qk_dot_by_d', False) else 1.0
    softmax_scale = 1.0 if not config.scale_attn_weights else head_dim ** (-attn_scale_power)
    softmax_scale *= getattr(config, 'mup_attn_multiplier', 1.0)
    if config.scale_attn_by_inverse_layer_idx:
        assert layer_idx is not None
        softmax_scale /= float(layer_idx + 1)
    dwconv = getattr(config, 'attn_dwconv', False)
    if dwconv:
        assert process_group is None, 'TensorParallel MHA does not support dwconv yet'
    qkv_proj_bias = getattr(config, 'qkv_proj_bias', True)
    out_proj_bias = getattr(config, 'out_proj_bias', True)
    rotary_emb_dim = int(getattr(config, 'rotary_emb_fraction', 0.0) * head_dim)
    rotary_emb_base = getattr(config, 'rotary_emb_base', 10000.0)
    rotary_emb_scale_base = getattr(config, 'rotary_emb_scale_base', None)
    rotary_emb_interleaved = getattr(config, 'rotary_emb_interleaved', False)
    use_alibi = getattr(config, 'use_alibi', False)
    window_size = getattr(config, 'window_size', (-1, -1))
    use_flash_attn = getattr(config, 'use_flash_attn', False)
    fused_bias_fc = getattr(config, 'fused_bias_fc', False)
    if not fused_bias_fc:
        assert process_group is None, 'TensorParallel MHA requires fused_bias_fc'
    mha_cls = MHA if process_group is None else ParallelMHA
    serial_kwargs = {'fused_bias_fc': fused_bias_fc, 'dwconv': dwconv} if process_group is None else {}
    parallel_kwargs = {'process_group': process_group, 'sequence_parallel': getattr(config, 'sequence_parallel', True)} if process_group is not None else {}
    num_heads_kv = getattr(config, 'n_head_kv', None)
    mixer_cls = partial(mha_cls, num_heads=config.num_attention_heads, num_heads_kv=num_heads_kv, qkv_proj_bias=qkv_proj_bias, out_proj_bias=out_proj_bias, dropout=config.attn_pdrop, softmax_scale=softmax_scale, causal=True, layer_idx=layer_idx, rotary_emb_dim=rotary_emb_dim, rotary_emb_base=rotary_emb_base, rotary_emb_scale_base=rotary_emb_scale_base, rotary_emb_interleaved=rotary_emb_interleaved, use_alibi=use_alibi, window_size=window_size, use_flash_attn=use_flash_attn, **serial_kwargs, **parallel_kwargs, **factory_kwargs)
    return mixer_cls