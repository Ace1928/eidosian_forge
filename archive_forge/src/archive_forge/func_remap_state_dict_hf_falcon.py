import math
import re
from collections import OrderedDict
import torch
import torch.nn.functional as F
from einops import rearrange
from transformers import FalconConfig, GPT2Config
def remap_state_dict_hf_falcon(state_dict, config):

    def key_mapping_layers(key):
        return re.sub('^transformer.h.', 'transformer.layers.', key)
    state_dict = OrderedDict(((key_mapping_layers(k), v) for k, v in state_dict.items()))

    def key_mapping_emb(key):
        return re.sub('^transformer.word_embeddings.', 'transformer.embeddings.word_embeddings.', key)
    state_dict = OrderedDict(((key_mapping_emb(k), v) for k, v in state_dict.items()))
    word_embeddings = state_dict.pop('transformer.embeddings.word_embeddings.weight')
    pad_vocab_size_multiple = getattr(config, 'pad_vocab_size_multiple', 1)
    vocab_size = math.ceil(config.vocab_size / pad_vocab_size_multiple) * pad_vocab_size_multiple
    state_dict['transformer.embeddings.word_embeddings.weight'] = F.pad(word_embeddings, (0, 0, 0, vocab_size - word_embeddings.shape[0]))
    if getattr(config, 'tie_word_embeddings'):
        state_dict['lm_head.weight'] = state_dict['transformer.embeddings.word_embeddings.weight']
    else:
        output_embeddings = state_dict.pop('lm_head.weight')
        state_dict['lm_head.weight'] = F.pad(output_embeddings, (0, 0, 0, vocab_size - output_embeddings.shape[0]))
        output_embeddings_bias = state_dict.pop('lm_head.bias')
        state_dict['lm_head.bias'] = F.pad(output_embeddings_bias, (0, vocab_size - output_embeddings_bias.shape[0]))

    def key_mapping_ln(key):
        key = re.sub('^transformer.layers.(\\d+).input_layernorm.', 'transformer.layers.\\1.norm1.', key)
        key = re.sub('^transformer.layers.(\\d+).post_attention_layernorm.', 'transformer.layers.\\1.norm2.', key)
        key = re.sub('^transformer.layers.(\\d+).ln_attn.', 'transformer.layers.\\1.norm1.', key)
        key = re.sub('^transformer.layers.(\\d+).ln_mlp.', 'transformer.layers.\\1.norm2.', key)
        return key
    state_dict = OrderedDict(((key_mapping_ln(k), v) for k, v in state_dict.items()))

    def key_mapping_mlp(key):
        key = re.sub('^transformer.layers.(\\d+).mlp.dense_h_to_4h.', 'transformer.layers.\\1.mlp.fc1.', key)
        key = re.sub('^transformer.layers.(\\d+).mlp.dense_4h_to_h.', 'transformer.layers.\\1.mlp.fc2.', key)
        return key
    state_dict = OrderedDict(((key_mapping_mlp(k), v) for k, v in state_dict.items()))

    def key_mapping_attn(key):
        key = re.sub('^transformer.layers.(\\d+).self_attention.query_key_value.', 'transformer.layers.\\1.mixer.Wqkv.', key)
        key = re.sub('^transformer.layers.(\\d+).self_attention.dense.', 'transformer.layers.\\1.mixer.out_proj.', key)
        return key
    state_dict = OrderedDict(((key_mapping_attn(k), v) for k, v in state_dict.items()))
    n_head = config.n_head
    n_head_kv = getattr(config, 'n_head_kv', 1)
    headdim = config.hidden_size // n_head
    for l in range(config.n_layer):
        Wqkv = rearrange(state_dict.pop(f'transformer.layers.{l}.mixer.Wqkv.weight'), '(group ratio headdim) ... -> group ratio headdim ...', ratio=n_head // n_head_kv + 2, headdim=headdim)
        Wq = rearrange(Wqkv[:, :-2], 'group ratio headdim ... -> (group ratio headdim) ...')
        Wk = rearrange(Wqkv[:, [-2]], 'group ratio headdim ... -> (group ratio headdim) ...')
        Wv = rearrange(Wqkv[:, [-1]], 'group ratio headdim ... -> (group ratio headdim) ...')
        state_dict[f'transformer.layers.{l}.mixer.Wqkv.weight'] = torch.cat([Wq, Wk, Wv], dim=0)
    return state_dict