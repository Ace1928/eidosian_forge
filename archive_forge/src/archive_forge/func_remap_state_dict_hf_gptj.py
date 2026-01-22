import math
import re
from collections import OrderedDict
import torch
import torch.nn.functional as F
from transformers import GPT2Config, GPTJConfig
def remap_state_dict_hf_gptj(state_dict, config):

    def key_mapping_layers(key):
        return re.sub('^transformer.h.', 'transformer.layers.', key)
    state_dict = OrderedDict(((key_mapping_layers(k), v) for k, v in state_dict.items()))

    def key_mapping_emb(key):
        return re.sub('^transformer.wte.', 'transformer.embeddings.word_embeddings.', key)
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
        return re.sub('^transformer.layers.(\\d+).ln_1.', 'transformer.layers.\\1.norm1.', key)
    state_dict = OrderedDict(((key_mapping_ln(k), v) for k, v in state_dict.items()))

    def key_mapping_mlp(key):
        key = re.sub('^transformer.layers.(\\d+).mlp.fc_in.', 'transformer.layers.\\1.mlp.fc1.', key)
        key = re.sub('^transformer.layers.(\\d+).mlp.fc_out.', 'transformer.layers.\\1.mlp.fc2.', key)
        return key
    state_dict = OrderedDict(((key_mapping_mlp(k), v) for k, v in state_dict.items()))
    for l in range(config.n_layer):
        Wq = state_dict.pop(f'transformer.layers.{l}.attn.q_proj.weight')
        Wk = state_dict.pop(f'transformer.layers.{l}.attn.k_proj.weight')
        Wv = state_dict.pop(f'transformer.layers.{l}.attn.v_proj.weight')
        state_dict[f'transformer.layers.{l}.mixer.Wqkv.weight'] = torch.cat([Wq, Wk, Wv], dim=0)
        state_dict.pop(f'transformer.layers.{l}.attn.bias')
        state_dict.pop(f'transformer.layers.{l}.attn.masked_bias')

    def key_mapping_attn(key):
        return re.sub('^transformer.layers.(\\d+).attn.out_proj.', 'transformer.layers.\\1.mixer.out_proj.', key)
    state_dict = OrderedDict(((key_mapping_attn(k), v) for k, v in state_dict.items()))
    return state_dict