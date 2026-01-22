import math
import re
from collections import OrderedDict
import torch
import torch.nn.functional as F
from transformers import GPT2Config, GPTBigCodeConfig, PretrainedConfig
def inv_remap_state_dict_hf_bigcode(state_dict, config: PretrainedConfig):
    """
    Map the state_dict of a flash_attn model to be Huggingface BigCode compatible.

    This function is meant to be the inverse of remap_state_dict_hf_bigcode.
    """

    def inv_key_mapping_pos_emb(key):
        return re.sub('^transformer.embeddings.position_embeddings.', 'transformer.wpe.', key)
    state_dict = OrderedDict(((inv_key_mapping_pos_emb(k), v) for k, v in state_dict.items()))
    word_embeddings = state_dict.pop('transformer.embeddings.word_embeddings.weight')
    word_embeddings = word_embeddings[:, :config.vocab_size]
    state_dict['transformer.wte.weight'] = word_embeddings
    state_dict['lm_head.weight'] = word_embeddings

    def inv_key_mapping_ln(key):
        key = re.sub('^transformer.ln_f.(weight|bias)', 'transformer.ln_f.\\1', key)
        key = re.sub('^transformer.layers.(\\d+).norm(1|2).(weight|bias)', 'transformer.h.\\1.ln_\\2.\\3', key)
        return key
    state_dict = OrderedDict(((inv_key_mapping_ln(k), v) for k, v in state_dict.items()))

    def inv_key_mapping_mlp(key):
        key = re.sub('^transformer.layers.(\\d+).mlp.fc1.weight', 'transformer.h.\\1.mlp.c_fc.weight', key)
        key = re.sub('^transformer.layers.(\\d+).mlp.fc2.weight', 'transformer.h.\\1.mlp.c_proj.weight', key)
        key = re.sub('^transformer.layers.(\\d+).mlp.fc1.bias', 'transformer.h.\\1.mlp.c_fc.bias', key)
        key = re.sub('^transformer.layers.(\\d+).mlp.fc2.bias', 'transformer.h.\\1.mlp.c_proj.bias', key)
        return key
    state_dict = OrderedDict(((inv_key_mapping_mlp(k), v) for k, v in state_dict.items()))
    for d in range(config.num_hidden_layers):
        embed_dim = config.n_embd
        head_dim = embed_dim // config.n_head
        Wqkv_weight = state_dict.pop(f'transformer.layers.{d}.mixer.Wqkv.weight')
        q, k, v = torch.split(Wqkv_weight, [embed_dim, head_dim * config.n_head, head_dim * config.n_head], dim=0)
        c_attn_weight = torch.cat((q, k[:head_dim], v[:head_dim]), dim=0)
        state_dict[f'transformer.h.{d}.attn.c_attn.weight'] = c_attn_weight
        Wqkv_bias = state_dict.pop(f'transformer.layers.{d}.mixer.Wqkv.bias')
        q, k, v = torch.split(Wqkv_bias, [embed_dim, head_dim * config.n_head, head_dim * config.n_head], dim=0)
        c_attn_bias = torch.cat((q, k[:head_dim], v[:head_dim]), dim=0)
        state_dict[f'transformer.h.{d}.attn.c_attn.bias'] = c_attn_bias

    def inv_key_mapping_attn(key):
        key = re.sub('^transformer.layers.(\\d+).mixer.out_proj.weight', 'transformer.h.\\1.attn.c_proj.weight', key)
        key = re.sub('^transformer.layers.(\\d+).mixer.out_proj.bias', 'transformer.h.\\1.attn.c_proj.bias', key)
        return key
    state_dict = OrderedDict(((inv_key_mapping_attn(k), v) for k, v in state_dict.items()))
    return state_dict