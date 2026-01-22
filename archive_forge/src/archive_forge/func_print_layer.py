import copy
import functools
import re
import weakref
import numpy as np
import tensorflow.compat.v2 as tf
from keras.src import initializers
from keras.src.utils import io_utils
from tensorflow.python.util.tf_export import keras_export
def print_layer(layer, nested_level=0, is_nested_last=False):
    if sequential_like:
        print_layer_summary(layer, nested_level)
    else:
        print_layer_summary_with_connections(layer, nested_level)
    if expand_nested and hasattr(layer, 'layers') and layer.layers:
        print_fn('|' * (nested_level + 1) + '¯' * (line_length - 2 * nested_level - 2) + '|' * (nested_level + 1))
        nested_layer = layer.layers
        is_nested_last = False
        for i in range(len(nested_layer)):
            if i == len(nested_layer) - 1:
                is_nested_last = True
            print_layer(nested_layer[i], nested_level + 1, is_nested_last)
        print_fn('|' * nested_level + '¯' * (line_length - 2 * nested_level) + '|' * nested_level)
    if not is_nested_last:
        print_fn('|' * nested_level + ' ' * (line_length - 2 * nested_level) + '|' * nested_level)