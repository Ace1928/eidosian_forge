import logging
import re
from collections import OrderedDict
from collections.abc import Sequence
from functools import partial
from typing import Any, Mapping
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers import BertConfig, PretrainedConfig
from transformers.models.bert.modeling_bert import (
from flash_attn.bert_padding import (
from flash_attn.modules.block import Block
from flash_attn.modules.embedding import BertEmbeddings
from flash_attn.modules.mha import MHA
from flash_attn.modules.mlp import FusedMLP, Mlp
from flash_attn.utils.pretrained import state_dict_from_pretrained
def inv_remap_state_dict(state_dict, config: PretrainedConfig):
    """
    Map the state_dict of a flash_attn model to be Huggingface BERT compatible.

    This function is meant to be the inverse of remap_state_dict.
    """
    pad_vocab_size_multiple = getattr(config, 'pad_vocab_size_multiple', 1)
    if pad_vocab_size_multiple > 1:
        word_embeddings = state_dict['bert.embeddings.word_embeddings.weight']
        decoder_weight = state_dict['cls.predictions.decoder.weight']
        decoder_bias = state_dict['cls.predictions.decoder.bias']
        state_dict['bert.embeddings.word_embeddings.weight'] = word_embeddings[:config.orig_vocab_size, :]
        state_dict['cls.predictions.decoder.weight'] = decoder_weight[:config.orig_vocab_size, :]
        state_dict['cls.predictions.decoder.bias'] = decoder_bias[:config.orig_vocab_size]
    for d in range(config.num_hidden_layers):
        last_layer_subset = getattr(config, 'last_layer_subset', False)
        if not last_layer_subset or d != config.num_hidden_layers - 1:
            Wqkv_weights = state_dict.pop(f'bert.encoder.layers.{d}.mixer.Wqkv.weight')
            Wqkv_biases = state_dict.pop(f'bert.encoder.layers.{d}.mixer.Wqkv.bias')
            state_dict[f'bert.encoder.layers.{d}.attention.self.query.weight'] = Wqkv_weights[:Wqkv_weights.shape[0] // 3, :]
            state_dict[f'bert.encoder.layers.{d}.attention.self.key.weight'] = Wqkv_weights[Wqkv_weights.shape[0] // 3:2 * Wqkv_weights.shape[0] // 3, :]
            state_dict[f'bert.encoder.layers.{d}.attention.self.value.weight'] = Wqkv_weights[2 * Wqkv_weights.shape[0] // 3:, :]
            state_dict[f'bert.encoder.layers.{d}.attention.self.query.bias'] = Wqkv_biases[:Wqkv_biases.shape[0] // 3]
            state_dict[f'bert.encoder.layers.{d}.attention.self.key.bias'] = Wqkv_biases[Wqkv_biases.shape[0] // 3:2 * Wqkv_biases.shape[0] // 3]
            state_dict[f'bert.encoder.layers.{d}.attention.self.value.bias'] = Wqkv_biases[2 * Wqkv_biases.shape[0] // 3:]
        else:
            Wq_weight = state_dict.pop(f'bert.encoder.layers.{d}.mixer.Wq.weight')
            Wkv_weights = state_dict.pop(f'bert.encoder.layers.{d}.mixer.Wkv.weight')
            Wq_bias = state_dict.pop(f'bert.encoder.layers.{d}.mixer.Wq.bias')
            Wkv_biases = state_dict.pop(f'bert.encoder.layers.{d}.mixer.Wkv.bias')
            state_dict[f'bert.encoder.layers.{d}.attention.self.query.weight'] = Wq_weight
            state_dict[f'bert.encoder.layers.{d}.attention.self.key.weight'] = Wkv_weights[:Wkv_weights.shape[0] // 2, :]
            state_dict[f'bert.encoder.layers.{d}.attention.self.value.weight'] = Wkv_weights[Wkv_weights.shape[0] // 2:, :]
            state_dict[f'bert.encoder.layers.{d}.attention.self.query.bias'] = Wq_bias
            state_dict[f'bert.encoder.layers.{d}.attention.self.key.bias'] = Wkv_biases[:Wkv_biases.shape[0] // 2]
            state_dict[f'bert.encoder.layers.{d}.attention.self.value.bias'] = Wkv_biases[Wkv_biases.shape[0] // 2:]

    def inv_key_mapping_ln(key):
        key = re.sub('bert.emb_ln.', 'bert.embeddings.LayerNorm.', key)
        key = re.sub('bert.encoder.layers.(\\d+).norm1.(weight|bias)', 'bert.encoder.layers.\\1.attention.output.LayerNorm.\\2', key)
        key = re.sub('bert.encoder.layers.(\\d+).norm2.(weight|bias)', 'bert.encoder.layers.\\1.output.LayerNorm.\\2', key)
        key = re.sub('cls.predictions.transform.layer_norm.(weight|bias)', 'cls.predictions.transform.LayerNorm.\\1', key)
        return key

    def inv_key_mapping_ln_gamma_beta(key):
        key = re.sub('LayerNorm.weight$', 'LayerNorm.gamma', key)
        key = re.sub('LayerNorm.bias$', 'LayerNorm.beta', key)
        return key

    def inv_key_mapping_layers(key):
        return re.sub('bert.encoder.layers.', 'bert.encoder.layer.', key)

    def inv_key_mapping_mlp(key):
        key = re.sub('bert.encoder.layer.(\\d+).mlp.fc1.(weight|bias)', 'bert.encoder.layer.\\1.intermediate.dense.\\2', key)
        key = re.sub('bert.encoder.layer.(\\d+).mlp.fc2.(weight|bias)', 'bert.encoder.layer.\\1.output.dense.\\2', key)
        return key

    def inv_key_mapping_attn(key):
        return re.sub('bert.encoder.layer.(\\d+).mixer.out_proj.(weight|bias)', 'bert.encoder.layer.\\1.attention.output.dense.\\2', key)

    def inv_key_mapping_decoder_bias(key):
        return re.sub('cls.predictions.decoder.bias', 'cls.predictions.bias', key)
    state_dict = OrderedDict(((inv_key_mapping_ln(key), value) for key, value in state_dict.items()))
    state_dict = OrderedDict(((inv_key_mapping_ln_gamma_beta(key), value) for key, value in state_dict.items()))
    state_dict = OrderedDict(((inv_key_mapping_layers(key), value) for key, value in state_dict.items()))
    state_dict = OrderedDict(((inv_key_mapping_mlp(key), value) for key, value in state_dict.items()))
    state_dict = OrderedDict(((inv_key_mapping_attn(key), value) for key, value in state_dict.items()))
    state_dict = OrderedDict(((inv_key_mapping_decoder_bias(key), value) for key, value in state_dict.items()))
    return state_dict