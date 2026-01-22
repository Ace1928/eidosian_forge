import copy
import inspect
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from tensorflow.compiler.tf2xla.python.xla import dynamic_update_slice
from ..modeling_tf_outputs import TFCausalLMOutputWithPast, TFSeq2SeqLMOutput
from ..models.auto import (
from ..tf_utils import shape_list, stable_softmax
from ..utils import ModelOutput, logging
from .configuration_utils import GenerationConfig
from .tf_logits_process import (
def tf_top_k_top_p_filtering(logits, top_k=0, top_p=1.0, filter_value=-float('Inf'), min_tokens_to_keep=1):
    """
    Filter a distribution of logits using top-k and/or nucleus (top-p) filtering

    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        top_k (`int`, *optional*, defaults to 0):
            If > 0, only keep the top k tokens with highest probability (top-k filtering)
        top_p (`float`, *optional*, defaults to 1.0):
            If < 1.0, only keep the top tokens with cumulative probability >= top_p (nucleus filtering). Nucleus
            filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        min_tokens_to_keep (`int`, *optional*, defaults to 1):
            Minimumber of tokens we keep per batch example in the output.

    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    warnings.warn('`tf_top_k_top_p_filtering` is scheduled for deletion in v4.39. Use `TFTopKLogitsWarper` and `TFTopPLogitsWarper` instead.', DeprecationWarning)
    logits_shape = shape_list(logits)
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits_shape[-1])
        indices_to_remove = logits < tf.math.top_k(logits, k=top_k)[0][..., -1, None]
        logits = tf.where(indices_to_remove, filter_value, logits)
    if top_p < 1.0:
        sorted_indices = tf.argsort(logits, direction='DESCENDING')
        sorted_logits = tf.gather(logits, sorted_indices, axis=-1, batch_dims=1)
        cumulative_probs = tf.math.cumsum(stable_softmax(sorted_logits, axis=-1), axis=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            sorted_indices_to_remove = tf.concat([tf.zeros_like(sorted_indices_to_remove[:, :min_tokens_to_keep]), sorted_indices_to_remove[:, min_tokens_to_keep:]], -1)
        sorted_indices_to_remove = tf.concat([tf.zeros_like(sorted_indices_to_remove[:, :1]), sorted_indices_to_remove[:, :-1]], -1)
        indices_to_remove = scatter_values_on_batch_indices(sorted_indices_to_remove, sorted_indices)
        logits = tf.where(indices_to_remove, filter_value, logits)
    return logits