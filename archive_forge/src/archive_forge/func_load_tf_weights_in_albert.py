import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from .configuration_albert import AlbertConfig
def load_tf_weights_in_albert(model, config, tf_checkpoint_path):
    """Load tf checkpoints in a pytorch model."""
    try:
        import re
        import numpy as np
        import tensorflow as tf
    except ImportError:
        logger.error('Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see https://www.tensorflow.org/install/ for installation instructions.')
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    logger.info(f'Converting TensorFlow checkpoint from {tf_path}')
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        logger.info(f'Loading TF weight {name} with shape {shape}')
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)
    for name, array in zip(names, arrays):
        print(name)
    for name, array in zip(names, arrays):
        original_name = name
        name = name.replace('module/', '')
        name = name.replace('ffn_1', 'ffn')
        name = name.replace('bert/', 'albert/')
        name = name.replace('attention_1', 'attention')
        name = name.replace('transform/', '')
        name = name.replace('LayerNorm_1', 'full_layer_layer_norm')
        name = name.replace('LayerNorm', 'attention/LayerNorm')
        name = name.replace('transformer/', '')
        name = name.replace('intermediate/dense/', '')
        name = name.replace('ffn/intermediate/output/dense/', 'ffn_output/')
        name = name.replace('/output/', '/')
        name = name.replace('/self/', '/')
        name = name.replace('pooler/dense', 'pooler')
        name = name.replace('cls/predictions', 'predictions')
        name = name.replace('predictions/attention', 'predictions')
        name = name.replace('embeddings/attention', 'embeddings')
        name = name.replace('inner_group_', 'albert_layers/')
        name = name.replace('group_', 'albert_layer_groups/')
        if len(name.split('/')) == 1 and ('output_bias' in name or 'output_weights' in name):
            name = 'classifier/' + name
        if 'seq_relationship' in name:
            name = name.replace('seq_relationship/output_', 'sop_classifier/classifier/')
            name = name.replace('weights', 'weight')
        name = name.split('/')
        if 'adam_m' in name or 'adam_v' in name or 'AdamWeightDecayOptimizer' in name or ('AdamWeightDecayOptimizer_1' in name) or ('global_step' in name):
            logger.info(f'Skipping {'/'.join(name)}')
            continue
        pointer = model
        for m_name in name:
            if re.fullmatch('[A-Za-z]+_\\d+', m_name):
                scope_names = re.split('_(\\d+)', m_name)
            else:
                scope_names = [m_name]
            if scope_names[0] == 'kernel' or scope_names[0] == 'gamma':
                pointer = getattr(pointer, 'weight')
            elif scope_names[0] == 'output_bias' or scope_names[0] == 'beta':
                pointer = getattr(pointer, 'bias')
            elif scope_names[0] == 'output_weights':
                pointer = getattr(pointer, 'weight')
            elif scope_names[0] == 'squad':
                pointer = getattr(pointer, 'classifier')
            else:
                try:
                    pointer = getattr(pointer, scope_names[0])
                except AttributeError:
                    logger.info(f'Skipping {'/'.join(name)}')
                    continue
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]
        if m_name[-11:] == '_embeddings':
            pointer = getattr(pointer, 'weight')
        elif m_name == 'kernel':
            array = np.transpose(array)
        try:
            if pointer.shape != array.shape:
                raise ValueError(f'Pointer shape {pointer.shape} and array shape {array.shape} mismatched')
        except ValueError as e:
            e.args += (pointer.shape, array.shape)
            raise
        print(f'Initialize PyTorch weight {name} from {original_name}')
        pointer.data = torch.from_numpy(array)
    return model