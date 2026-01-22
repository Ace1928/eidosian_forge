import argparse
import collections
import numpy as np
import torch
from flax import traverse_util
from t5x import checkpoints
from transformers import MT5Config, UMT5EncoderModel, UMT5ForConditionalGeneration
from transformers.utils import logging
def convert_t5x_to_pytorch(variables: dict, *, num_layers: int, is_encoder_only: bool, scalable_attention: bool=False):
    """Converts the parameters from T5X-Flax to Transformers-PyTorch."""
    old = traverse_util.flatten_dict(variables['target'])
    old = {'/'.join(k): v for k, v in old.items()}
    split_mlp_wi = 'encoder/encoder/mlp/wi_0/kernel' in old
    print('Split MLP:', split_mlp_wi)
    new = collections.OrderedDict()
    new['shared.weight'] = old['token_embedder/embedding']
    for i in range(num_layers):
        layer_norm = t5x_layer_norm_lookup(old, i, 'encoder', 'pre_attention_layer_norm')
        k, o, q, v = t5x_attention_lookup(old, i, 'encoder', 'attention')
        new[f'encoder.block.{i}.layer.0.layer_norm.weight'] = layer_norm
        new[f'encoder.block.{i}.layer.0.SelfAttention.k.weight'] = k.T
        new[f'encoder.block.{i}.layer.0.SelfAttention.o.weight'] = o.T
        new[f'encoder.block.{i}.layer.0.SelfAttention.q.weight'] = q.T
        new[f'encoder.block.{i}.layer.0.SelfAttention.v.weight'] = v.T
        layer_norm = t5x_layer_norm_lookup(old, i, 'encoder', 'pre_mlp_layer_norm')
        wi, wo = t5x_mlp_lookup(old, i, 'encoder', split_mlp_wi)
        new[f'encoder.block.{i}.layer.1.layer_norm.weight'] = layer_norm
        if split_mlp_wi:
            new[f'encoder.block.{i}.layer.1.DenseReluDense.wi_0.weight'] = wi[0].T
            new[f'encoder.block.{i}.layer.1.DenseReluDense.wi_1.weight'] = wi[1].T
        else:
            new[f'encoder.block.{i}.layer.1.DenseReluDense.wi.weight'] = wi.T
        new[f'encoder.block.{i}.layer.1.DenseReluDense.wo.weight'] = wo.T
        if scalable_attention:
            new[f'encoder.block.{i}.layer.0.SelfAttention.relative_attention_bias.weight'] = t5x_relpos_bias_lookup(old, i, 'encoder').T
    new['encoder.final_layer_norm.weight'] = old['encoder/encoder_norm/scale']
    if not scalable_attention:
        new['encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight'] = t5x_relpos_bias_lookup(old, 0, 'encoder').T
        new['decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight'] = t5x_relpos_bias_lookup(old, 0, 'decoder').T
    if not is_encoder_only:
        for i in range(num_layers):
            layer_norm = t5x_layer_norm_lookup(old, i, 'decoder', 'pre_self_attention_layer_norm')
            k, o, q, v = t5x_attention_lookup(old, i, 'decoder', 'self_attention')
            new[f'decoder.block.{i}.layer.0.layer_norm.weight'] = layer_norm
            new[f'decoder.block.{i}.layer.0.SelfAttention.k.weight'] = k.T
            new[f'decoder.block.{i}.layer.0.SelfAttention.o.weight'] = o.T
            new[f'decoder.block.{i}.layer.0.SelfAttention.q.weight'] = q.T
            new[f'decoder.block.{i}.layer.0.SelfAttention.v.weight'] = v.T
            layer_norm = t5x_layer_norm_lookup(old, i, 'decoder', 'pre_cross_attention_layer_norm')
            k, o, q, v = t5x_attention_lookup(old, i, 'decoder', 'encoder_decoder_attention')
            new[f'decoder.block.{i}.layer.1.layer_norm.weight'] = layer_norm
            new[f'decoder.block.{i}.layer.1.EncDecAttention.k.weight'] = k.T
            new[f'decoder.block.{i}.layer.1.EncDecAttention.o.weight'] = o.T
            new[f'decoder.block.{i}.layer.1.EncDecAttention.q.weight'] = q.T
            new[f'decoder.block.{i}.layer.1.EncDecAttention.v.weight'] = v.T
            layer_norm = t5x_layer_norm_lookup(old, i, 'decoder', 'pre_mlp_layer_norm')
            wi, wo = t5x_mlp_lookup(old, i, 'decoder', split_mlp_wi)
            new[f'decoder.block.{i}.layer.2.layer_norm.weight'] = layer_norm
            if split_mlp_wi:
                new[f'decoder.block.{i}.layer.2.DenseReluDense.wi_0.weight'] = wi[0].T
                new[f'decoder.block.{i}.layer.2.DenseReluDense.wi_1.weight'] = wi[1].T
            else:
                new[f'encoder.block.{i}.layer.2.DenseReluDense.wi.weight'] = wi.T
            new[f'decoder.block.{i}.layer.2.DenseReluDense.wo.weight'] = wo.T
            if scalable_attention:
                new[f'decoder.block.{i}.layer.0.SelfAttention.relative_attention_bias.weight'] = t5x_relpos_bias_lookup(old, i, 'decoder').T
        new['decoder.final_layer_norm.weight'] = old['decoder/decoder_norm/scale']
        if 'decoder/logits_dense/kernel' in old:
            new['lm_head.weight'] = old['decoder/logits_dense/kernel'].T
    return new