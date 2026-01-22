import numpy as np
from . import _translation_utils as translation_utils
from .... import symbol
def reduce_log_sum_exp(attrs, inputs, proto_obj):
    """Reduce the array along a given axis by log sum exp value"""
    keep_dims = True if 'keepdims' not in attrs else attrs.get('keepdims')
    exp_op = symbol.exp(inputs[0])
    sum_op = symbol.sum(exp_op, axis=attrs.get('axes'), keepdims=keep_dims)
    log_sym = symbol.log(sum_op)
    return (log_sym, attrs, inputs)