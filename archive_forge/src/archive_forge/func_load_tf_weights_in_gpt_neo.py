import os
from typing import Optional, Tuple, Union
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import is_torch_greater_or_equal_than_1_13
from ...utils import (
from .configuration_gpt_neo import GPTNeoConfig
def load_tf_weights_in_gpt_neo(model, config, gpt_neo_checkpoint_path):
    """Load tf checkpoints in a pytorch model"""
    try:
        import re
        import tensorflow as tf
    except ImportError:
        logger.error('Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see https://www.tensorflow.org/install/ for installation instructions.')
        raise
    tf_path = os.path.abspath(gpt_neo_checkpoint_path)
    logger.info(f'Converting TensorFlow checkpoint from {tf_path}')
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        if 'global_step' not in name and 'adam' not in name:
            array = tf.train.load_variable(tf_path, name)
            array = tf.dtypes.cast(array.squeeze(), tf.float32).numpy()
            name = name.replace('attn/q', 'attn/attention/q_proj/w')
            name = name.replace('attn/k', 'attn/attention/k_proj/w')
            name = name.replace('attn/v', 'attn/attention/v_proj/w')
            name = name.replace('attn/o', 'attn/attention/out_proj/w')
            name = name.replace('norm_1', 'ln_1')
            name = name.replace('norm_2', 'ln_2')
            name = name.replace('attn/compute_output_bias/o_b', 'attn/attention/out_proj/b')
            name = name.replace('conv1d_main/c_fc/kernel', 'c_fc/w')
            name = name.replace('conv1d_main/c_fc/bias', 'c_fc/b')
            name = name.replace('conv1d_main/c_proj/kernel', 'c_proj/w')
            name = name.replace('conv1d_main/c_proj/bias', 'c_proj/b')
            names.append(name)
            arrays.append(array)
    for name, array in zip(names, arrays):
        name = name[5:]
        name = name.split('/')
        pointer = model.transformer
        for m_name in name:
            if re.fullmatch('[A-Za-z]+\\d+', m_name):
                scope_names = re.split('(\\d+)', m_name)
            else:
                scope_names = [m_name]
            if scope_names[0] == 'w' or scope_names[0] == 'g':
                pointer = getattr(pointer, 'weight')
            elif scope_names[0] == 'b':
                pointer = getattr(pointer, 'bias')
            elif scope_names[0] == 'wpe' or scope_names[0] == 'wte':
                pointer = getattr(pointer, scope_names[0])
                pointer = getattr(pointer, 'weight')
            else:
                pointer = getattr(pointer, scope_names[0])
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]
        if name[-1] == 'w' and name[-2] in ['out_proj', 'k_proj', 'q_proj', 'v_proj', 'c_proj', 'c_fc']:
            array = array.transpose()
        if name == ['wte']:
            array = array[:config.vocab_size]
        if pointer.shape != array.shape:
            raise ValueError(f'Pointer shape {pointer.shape} and array shape {array.shape} mismatched {name}')
        print(f'Initialize PyTorch weight {name}')
        pointer.data = torch.from_numpy(array)
    embs = model.transformer.wte.weight
    lin = nn.Linear(embs.size()[1], embs.size()[0], bias=False)
    lin.weight = embs
    model.set_output_embeddings(lin)
    return model