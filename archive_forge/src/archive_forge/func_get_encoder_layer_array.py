import argparse
import tensorflow as tf
import torch
from transformers import BertConfig, BertForMaskedLM
from transformers.models.bert.modeling_bert import (
from transformers.utils import logging
def get_encoder_layer_array(layer_index: int, name: str):
    full_name = f'encoder/_transformer_layers/{layer_index}/{name}/.ATTRIBUTES/VARIABLE_VALUE'
    array = tf.train.load_variable(tf_checkpoint_path, full_name)
    if 'kernel' in name:
        array = array.transpose()
    return torch.from_numpy(array)