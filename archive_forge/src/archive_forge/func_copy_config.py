import argparse
import os
from os.path import isfile
import torch
from transformers import GPT2Config
def copy_config(config_hf, config_megatron):
    """Copy the config from Megatron to hf."""
    config_hf.vocab_size = 64000
    config_hf.n_positions = config_megatron['encoder_seq_length']
    config_hf.n_embd = config_megatron['hidden_size']
    config_hf.n_layer = config_megatron['num_layers']
    config_hf.n_head = config_megatron['num_attention_heads']
    config_hf.n_inner = config_megatron['ffn_hidden_size']
    config_hf.activation_function = 'gelu'
    config_hf.resid_pdrop = 0.1
    config_hf.embd_pdrop = 0.1
    config_hf.attn_pdrop = 0.1
    config_hf.layer_norm_epsilon = config_megatron['layernorm_epsilon']
    config_hf.initializer_range = config_megatron['init_method_std']
    config_hf.apply_query_key_layer_scaling = config_megatron['apply_query_key_layer_scaling']
    config_hf.normalize_attention_scores = True
    config_hf.use_cache = True
    if config_megatron['hidden_size'] == 4096:
        config_hf.bos_token_id = 1
        config_hf.eos_token_id = 1
        config_hf.pad_token_id = 0
    else:
        config_hf.bos_token_id = 2
        config_hf.eos_token_id = 3
        config_hf.pad_token_id = 0
    return config_hf