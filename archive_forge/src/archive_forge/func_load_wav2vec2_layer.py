import argparse
import json
import os
import fairseq
import torch
from fairseq.data import Dictionary
from transformers import (
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2ForSequenceClassification
def load_wav2vec2_layer(name, value, hf_model=None, hf_dict=None):
    is_used = False
    for key, mapped_key in MAPPING.items():
        mapped_key = 'wav2vec2.' + mapped_key if mapped_key not in TOP_LEVEL_KEYS else mapped_key
        if key in name or key.split('w2v_model.')[-1] == name.split('.')[0]:
            is_used = True
            if '*' in mapped_key:
                layer_index = name.split(key)[0].split('.')[-2]
                mapped_key = mapped_key.replace('*', layer_index)
            if 'weight_g' in name:
                weight_type = 'weight_g'
            elif 'weight_v' in name:
                weight_type = 'weight_v'
            elif 'bias' in name:
                weight_type = 'bias'
            elif 'weight' in name:
                weight_type = 'weight'
            else:
                weight_type = None
            if hf_dict is not None:
                rename_dict(mapped_key, value, name, weight_type, hf_dict)
            else:
                set_recursively(mapped_key, value, name, weight_type, hf_model)
            return is_used
    return is_used