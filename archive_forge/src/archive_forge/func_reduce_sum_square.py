import numpy as np
from . import _translation_utils as translation_utils
from .... import symbol
def reduce_sum_square(attrs, inputs, proto_obj):
    """Reduce the array along a given axis by sum square value"""
    square_op = symbol.square(inputs[0])
    sum_op = symbol.sum(square_op, axis=attrs.get('axes'), keepdims=attrs.get('keepdims'))
    return (sum_op, attrs, inputs)