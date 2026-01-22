import argparse
from typing import Dict
import tensorflow as tf
import torch
from tqdm import tqdm
from transformers import BigBirdPegasusConfig, BigBirdPegasusForConditionalGeneration
def convert_bigbird_pegasus(tf_weights: dict, config_update: dict) -> BigBirdPegasusForConditionalGeneration:
    cfg = BigBirdPegasusConfig(**config_update)
    torch_model = BigBirdPegasusForConditionalGeneration(cfg)
    state_dict = torch_model.state_dict()
    mapping = {}
    decoder_weights = {k: tf_weights[k] for k in tf_weights if k.startswith('pegasus/decoder')}
    remaining_weights = {k: tf_weights[k] for k in tf_weights if not k.startswith('pegasus/decoder')}
    for k, v in tqdm(decoder_weights.items(), 'tf -> hf conversion'):
        conditions = [k.endswith(ending) for ending in KEYS_TO_IGNORE]
        if any(conditions):
            continue
        patterns = DECODER_PATTERNS
        new_k = rename_state_dict_key(k, patterns)
        if new_k not in state_dict:
            raise ValueError(f'could not find new key {new_k} in state dict. (converted from {k})')
        if any((True if i in k else False for i in ['dense', 'query', 'key', 'value'])):
            v = v.T
        mapping[new_k] = torch.from_numpy(v)
        assert v.shape == state_dict[new_k].shape, f'{new_k}, {k}, {v.shape}, {state_dict[new_k].shape}'
    for k, v in tqdm(remaining_weights.items(), 'tf -> hf conversion'):
        conditions = [k.endswith(ending) for ending in KEYS_TO_IGNORE]
        if any(conditions):
            continue
        patterns = REMAINING_PATTERNS
        new_k = rename_state_dict_key(k, patterns)
        if new_k not in state_dict and k != 'pegasus/embeddings/position_embeddings':
            raise ValueError(f'could not find new key {new_k} in state dict. (converted from {k})')
        if any((True if i in k else False for i in ['dense', 'query', 'key', 'value'])):
            v = v.T
        mapping[new_k] = torch.from_numpy(v)
        if k != 'pegasus/embeddings/position_embeddings':
            assert v.shape == state_dict[new_k].shape, f'{new_k}, {k}, {v.shape}, {state_dict[new_k].shape}'
    mapping['model.encoder.embed_positions.weight'] = mapping['model.embed_positions.weight']
    mapping['model.decoder.embed_positions.weight'] = mapping.pop('model.embed_positions.weight')
    missing, extra = torch_model.load_state_dict(mapping, strict=False)
    unexpected_missing = [k for k in missing if k not in ['final_logits_bias', 'model.encoder.embed_tokens.weight', 'model.decoder.embed_tokens.weight', 'lm_head.weight']]
    assert unexpected_missing == [], f'no matches found for the following torch keys {unexpected_missing}'
    assert extra == [], f'no matches found for the following tf keys {extra}'
    return torch_model