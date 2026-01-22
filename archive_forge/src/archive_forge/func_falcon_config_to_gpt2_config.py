import math
import re
from collections import OrderedDict
import torch
import torch.nn.functional as F
from einops import rearrange
from transformers import FalconConfig, GPT2Config
def falcon_config_to_gpt2_config(falcon_config: FalconConfig) -> GPT2Config:
    n_head_kv = getattr(falcon_config, 'n_head_kv', 1 if getattr(falcon_config, 'multi_query', False) else falcon_config.n_head)
    parallel_block_tied_norm = n_head_kv == 1
    return GPT2Config(vocab_size=falcon_config.vocab_size, n_positions=0, n_embd=falcon_config.hidden_size, n_layer=falcon_config.n_layer, n_head=falcon_config.n_head, n_inner=falcon_config.hidden_size * 4, activation_function='gelu', resid_pdrop=falcon_config.hidden_dropout, embd_pdrop=0.0, attn_pdrop=falcon_config.attention_dropout, layer_norm_epsilon=falcon_config.layer_norm_epsilon, initializer_range=falcon_config.initializer_range, bos_token_id=falcon_config.bos_token_id, eos_token_id=falcon_config.eos_token_id, parallel_block=falcon_config.parallel_attn, n_head_kv=n_head_kv, parallel_block_tied_norm=parallel_block_tied_norm, rotary_emb_fraction=1.0, rotary_emb_interleaved=False, tie_word_embeddings=True, qkv_proj_bias=falcon_config.bias, out_proj_bias=falcon_config.bias, mlp_fc1_bias=falcon_config.bias, mlp_fc2_bias=falcon_config.bias, lm_head_bias=False)