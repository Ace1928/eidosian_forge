import json
import math
import os
import re
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Union
import torch
import torch.nn.functional as F
from sentencepiece import SentencePieceProcessor
from transformers import GPT2Config, LlamaConfig
from einops import rearrange
def inv_remap_state_dict_hf_llama(state_dict: Dict[str, torch.Tensor], config: GPT2Config) -> Dict[str, torch.Tensor]:
    """Convert the state_dict in standard GPT format to Hugging Face format.

    This function is meant to be the inverse of remap_state_dict_hf_llama, up to a
    multiplier pad in the embedding and lm_head. That is if the original embedding
    isn't a multiple of pad_vocab_size_multiple, then
    inv_remap_state_dict_hf_llama(remap_state_dict_hf_llama(state_dict)) != state_dict.

    This function modifies state_dict in place.
    """

    def key_mapping_emb(key):
        return re.sub('^transformer.embeddings.word_embeddings.', 'model.embed_tokens.', key)
    state_dict = OrderedDict(((key_mapping_emb(k), v) for k, v in state_dict.items()))
    word_embeddings = state_dict.pop('model.embed_tokens.weight')
    pad_vocab_size_multiple = getattr(config, 'pad_vocab_size_multiple', 1)
    vocab_size = math.ceil(word_embeddings.shape[0] / pad_vocab_size_multiple) * pad_vocab_size_multiple
    state_dict['model.embed_tokens.weight'] = F.pad(word_embeddings, (0, 0, 0, vocab_size - word_embeddings.shape[0]))
    if getattr(config, 'tie_word_embeddings'):
        state_dict['lm_head.weight'] = state_dict['model.embed_tokens.weight']
    else:
        output_embeddings = state_dict.pop('lm_head.weight')
        vocab_size = math.ceil(output_embeddings.shape[0] / pad_vocab_size_multiple) * pad_vocab_size_multiple
        state_dict['lm_head.weight'] = F.pad(output_embeddings, (0, 0, 0, vocab_size - output_embeddings.shape[0]))
    for l in range(config.n_layer):
        w3, w1 = torch.chunk(state_dict.pop(f'transformer.layers.{l}.mlp.fc1.weight'), chunks=2, dim=0)
        state_dict[f'model.layers.{l}.mlp.gate_proj.weight'] = w1
        state_dict[f'model.layers.{l}.mlp.up_proj.weight'] = w3

    def key_mapping_mlp(key):
        return re.sub('^transformer.layers.(\\d+).mlp.fc2.', 'model.layers.\\1.mlp.down_proj.', key)
    state_dict = OrderedDict(((key_mapping_mlp(k), v) for k, v in state_dict.items()))

    def key_mapping_ln(key):
        key = re.sub('^transformer.ln_f.', 'model.norm.', key)
        key = re.sub('^transformer.layers.(\\d+).norm1.', 'model.layers.\\1.input_layernorm.', key)
        key = re.sub('^transformer.layers.(\\d+).norm2.', 'model.layers.\\1.post_attention_layernorm.', key)
        return key
    state_dict = OrderedDict(((key_mapping_ln(k), v) for k, v in state_dict.items()))

    def permute(w):
        return rearrange(w, '(h d two) n -> (h two d) n', d=config.n_embd // config.n_head // 2, two=2)
    n_head = config.n_head
    n_head_kv = getattr(config, 'n_head_kv', n_head)
    embed_dim = config.hidden_size
    head_dim = embed_dim // n_head
    q_dim = n_head * head_dim
    k_dim = v_dim = n_head_kv * head_dim
    for l in range(config.n_layer):
        Wqkv = state_dict.pop(f'transformer.layers.{l}.mixer.Wqkv.weight')
        Wq = Wqkv[:q_dim]
        Wk = Wqkv[q_dim:q_dim + k_dim]
        Wv = Wqkv[q_dim + k_dim:q_dim + k_dim + v_dim]
        state_dict[f'model.layers.{l}.self_attn.q_proj.weight'] = permute(Wq)
        state_dict[f'model.layers.{l}.self_attn.k_proj.weight'] = permute(Wk)
        state_dict[f'model.layers.{l}.self_attn.v_proj.weight'] = Wv
        state_dict.pop(f'transformer.layers.{l}.attention.inner_attention.rope.freqs', None)

    def key_mapping_attn(key):
        return re.sub('^transformer.layers.(\\d+).mixer.out_proj.', 'model.layers.\\1.self_attn.o_proj.', key)
    state_dict = OrderedDict(((key_mapping_attn(k), v) for k, v in state_dict.items()))
    return state_dict