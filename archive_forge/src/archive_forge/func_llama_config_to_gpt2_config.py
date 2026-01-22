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
def llama_config_to_gpt2_config(llama_config: LlamaConfig) -> GPT2Config:
    return GPT2Config(vocab_size=llama_config.vocab_size, n_positions=0, n_embd=llama_config.hidden_size, n_layer=llama_config.num_hidden_layers, n_head=llama_config.num_attention_heads, n_inner=llama_config.intermediate_size, activation_function='swiglu', resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0, layer_norm_epsilon=llama_config.rms_norm_eps, initializer_range=llama_config.initializer_range, bos_token_id=llama_config.bos_token_id, eos_token_id=llama_config.eos_token_id, pad_token_id=llama_config.pad_token_id, rms_norm=True, rotary_emb_fraction=1.0, rotary_emb_interleaved=True, tie_word_embeddings=False, qkv_proj_bias=False, out_proj_bias=False, mlp_fc1_bias=False, mlp_fc2_bias=False, rotary_emb_base=getattr(llama_config, 'rotary_emb_base', 10000.0), n_head_kv=llama_config.num_key_value_heads)