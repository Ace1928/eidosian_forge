import math
import json
import re
from pathlib import Path
from collections import OrderedDict
import torch
import torch.nn.functional as F
from einops import rearrange
from transformers import GPT2Config, AutoConfig, PretrainedConfig
def baichuan_config_to_gpt2_config(baichuan_config: PretrainedConfig) -> GPT2Config:
    use_rotary = baichuan_config.hidden_size < 5000
    return GPT2Config(vocab_size=baichuan_config.vocab_size, n_positions=0, n_embd=baichuan_config.hidden_size, n_layer=baichuan_config.num_hidden_layers, n_head=baichuan_config.num_attention_heads, n_inner=baichuan_config.intermediate_size, activation_function='swiglu', resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0, layer_norm_epsilon=baichuan_config.rms_norm_eps, initializer_range=baichuan_config.initializer_range, bos_token_id=baichuan_config.bos_token_id, eos_token_id=baichuan_config.eos_token_id, pad_token_id=baichuan_config.pad_token_id, rms_norm=True, rotary_emb_fraction=1.0 if use_rotary else 0.0, rotary_emb_interleaved=False, use_alibi=not use_rotary, use_flash_attn=not use_rotary, tie_word_embeddings=False, norm_head=baichuan_config.vocab_size > 70000, qkv_proj_bias=False, out_proj_bias=False, mlp_fc1_bias=False, mlp_fc2_bias=False)