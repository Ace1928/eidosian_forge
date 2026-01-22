import re
import string
import numpy as np
from keras.src import activations
from keras.src import backend
from keras.src import constraints
from keras.src import dtype_policies
from keras.src import initializers
from keras.src import ops
from keras.src import quantizers
from keras.src import regularizers
from keras.src.api_export import keras_export
from keras.src.layers.input_spec import InputSpec
from keras.src.layers.layer import Layer
def get_specs(equation, input_shape):
    possible_labels = string.ascii_letters
    dot_replaced_string = re.sub('\\.\\.\\.', '0', equation)
    split_string = re.match('([a-zA-Z]+),([a-zA-Z]+)->([a-zA-Z]+)', dot_replaced_string)
    if split_string is not None:
        input_spec = split_string.group(1)
        weight_spec = split_string.group(2)
        output_spec = split_string.group(3)
        return (input_spec, weight_spec, output_spec)
    split_string = re.match('0([a-zA-Z]+),([a-zA-Z]+)->0([a-zA-Z]+)', dot_replaced_string)
    if split_string is not None:
        input_spec = split_string.group(1)
        weight_spec = split_string.group(2)
        output_spec = split_string.group(3)
        elided = len(input_shape) - len(input_spec)
        possible_labels = sorted(set(possible_labels) - set(input_spec) - set(weight_spec) - set(output_spec))
        for i in range(elided):
            input_spec = possible_labels[i] + input_spec
            output_spec = possible_labels[i] + output_spec
        return (input_spec, weight_spec, output_spec)
    split_string = re.match('([a-zA-Z]{2,})0,([a-zA-Z]+)->([a-zA-Z]+)0', dot_replaced_string)
    if split_string is not None:
        input_spec = split_string.group(1)
        weight_spec = split_string.group(2)
        output_spec = split_string.group(3)
        elided = len(input_shape) - len(input_spec)
        possible_labels = sorted(set(possible_labels) - set(input_spec) - set(weight_spec) - set(output_spec))
        for i in range(elided):
            input_spec = input_spec + possible_labels[i]
            output_spec = output_spec + possible_labels[i]
        return (input_spec, weight_spec, output_spec)
    raise ValueError(f"Invalid einsum equation '{equation}'. Equations must be in the form [X],[Y]->[Z], ...[X],[Y]->...[Z], or [X]...,[Y]->[Z]....")