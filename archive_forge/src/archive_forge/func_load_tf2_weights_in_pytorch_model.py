import os
import re
import numpy
from .utils import ExplicitEnum, expand_dims, is_numpy_array, is_torch_tensor, logging, reshape, squeeze, tensor_size
from .utils import transpose as transpose_func
def load_tf2_weights_in_pytorch_model(pt_model, tf_weights, allow_missing_keys=False, output_loading_info=False):
    """Load TF2.0 symbolic weights in a PyTorch model"""
    try:
        import tensorflow as tf
        import torch
    except ImportError:
        logger.error('Loading a TensorFlow model in PyTorch, requires both PyTorch and TensorFlow to be installed. Please see https://pytorch.org/ and https://www.tensorflow.org/install/ for installation instructions.')
        raise
    tf_state_dict = {tf_weight.name: tf_weight.numpy() for tf_weight in tf_weights}
    return load_tf2_state_dict_in_pytorch_model(pt_model, tf_state_dict, allow_missing_keys=allow_missing_keys, output_loading_info=output_loading_info)