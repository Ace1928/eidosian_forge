import math
import re
from collections import OrderedDict
import torch
import torch.nn.functional as F
from einops import rearrange
from transformers import GPT2Config, GPTNeoXConfig
def gpt_neox_config_to_gpt2_config(gpt_neox_config: GPTNeoXConfig) -> GPT2Config:
    assert gpt_neox_config.rotary_emb_base == 10000
    return GPT2Config(vocab_size=gpt_neox_config.vocab_size, n_positions=0, n_embd=gpt_neox_config.hidden_size, n_layer=gpt_neox_config.num_hidden_layers, n_head=gpt_neox_config.num_attention_heads, n_inner=gpt_neox_config.intermediate_size, activation_function=gpt_neox_config.hidden_act, resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0, layer_norm_epsilon=gpt_neox_config.layer_norm_eps, initializer_range=gpt_neox_config.initializer_range, bos_token_id=gpt_neox_config.bos_token_id, eos_token_id=gpt_neox_config.eos_token_id, prenorm=True, parallel_block=gpt_neox_config.use_parallel_residual, parallel_block_tied_norm=False, rotary_emb_fraction=gpt_neox_config.rotary_pct, tie_word_embeddings=gpt_neox_config.tie_word_embeddings)