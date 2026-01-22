import numpy as np
from . import _translation_utils as translation_utils
from .... import symbol
def reduce_log_sum(attrs, inputs, proto_obj):
    """Reduce the array along a given axis by log sum value"""
    keep_dims = True if 'keepdims' not in attrs else attrs.get('keepdims')
    sum_op = symbol.sum(inputs[0], axis=attrs.get('axes'), keepdims=keep_dims)
    log_sym = symbol.log(sum_op)
    return (log_sym, attrs, inputs)