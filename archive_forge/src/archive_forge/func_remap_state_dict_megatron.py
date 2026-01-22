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
def remap_state_dict_megatron(state_dict, config):

    def key_mapping_transformer(key):
        key = re.sub('^language_model.encoder.', 'transformer.', key)
        key = re.sub('^language_model.', 'transformer.', key)
        return key
    state_dict = OrderedDict(((key_mapping_transformer(k), v) for k, v in state_dict.items()))

    def key_mapping_pos_emb(key):
        return re.sub('^wpe.', 'transformer.embeddings.position_embeddings.', key)
    state_dict = OrderedDict(((key_mapping_pos_emb(k), v) for k, v in state_dict.items()))
    word_embeddings = state_dict.pop('transformer.embedding.word_embeddings.weight')
    pad_vocab_size_multiple = getattr(config, 'pad_vocab_size_multiple', 1)
    vocab_size = math.ceil(word_embeddings.shape[0] / pad_vocab_size_multiple) * pad_vocab_size_multiple
    state_dict['transformer.embeddings.word_embeddings.weight'] = F.pad(word_embeddings, (0, 0, 0, vocab_size - word_embeddings.shape[0]))
    state_dict['lm_head.weight'] = state_dict['transformer.embeddings.word_embeddings.weight']

    def key_mapping_ln(key):
        key = re.sub('^transformer.final_layernorm.(weight|bias)', 'transformer.ln_f.\\1', key)
        key = re.sub('^transformer.layers.(\\d+).input_layernorm.(weight|bias)', 'transformer.layers.\\1.norm1.\\2', key)
        key = re.sub('^transformer.layers.(\\d+).post_attention_layernorm.(weight|bias)', 'transformer.layers.\\1.norm2.\\2', key)
        return key
    state_dict = OrderedDict(((key_mapping_ln(k), v) for k, v in state_dict.items()))

    def key_mapping_mlp(key):
        key = re.sub('^transformer.layers.(\\d+).mlp.dense_h_to_4h.(weight|bias)', 'transformer.layers.\\1.mlp.fc1.\\2', key)
        key = re.sub('^transformer.layers.(\\d+).mlp.dense_4h_to_h.(weight|bias)', 'transformer.layers.\\1.mlp.fc2.\\2', key)
        return key
    state_dict = OrderedDict(((key_mapping_mlp(k), v) for k, v in state_dict.items()))

    def key_mapping_attn(key):
        key = re.sub('^transformer.layers.(\\d+).self_attention.rotary_emb.inv_freq', 'transformer.layers.\\1.mixer.rotary_emb.inv_freq', key)
        key = re.sub('^transformer.layers.(\\d+).self_attention.query_key_value.(weight|bias)', 'transformer.layers.\\1.mixer.Wqkv.\\2', key)
        key = re.sub('^transformer.layers.(\\d+).self_attention.dense.(weight|bias)', 'transformer.layers.\\1.mixer.out_proj.\\2', key)
        return key
    state_dict = OrderedDict(((key_mapping_attn(k), v) for k, v in state_dict.items()))
    headdim = config.hidden_size // config.num_attention_heads
    for d in range(config.num_hidden_layers):
        Wqkv = state_dict.pop(f'transformer.layers.{d}.mixer.Wqkv.weight')
        state_dict[f'transformer.layers.{d}.mixer.Wqkv.weight'] = rearrange(Wqkv, '(nheads three headdim) ... -> (three nheads headdim) ...', three=3, headdim=headdim)
        bqkv = state_dict.pop(f'transformer.layers.{d}.mixer.Wqkv.bias')
        state_dict[f'transformer.layers.{d}.mixer.Wqkv.bias'] = rearrange(bqkv, '(nheads three headdim) -> (three nheads headdim)', three=3, headdim=headdim)
    return state_dict