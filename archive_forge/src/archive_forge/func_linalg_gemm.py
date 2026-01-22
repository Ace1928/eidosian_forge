import numpy as np
from . import _translation_utils as translation_utils
from .... import symbol
def linalg_gemm(attrs, inputs, proto_obj):
    """Performs general matrix multiplication and accumulation"""
    trans_a = 0
    trans_b = 0
    alpha = 1
    beta = 1
    if 'transA' in attrs:
        trans_a = attrs['transA']
    if 'transB' in attrs:
        trans_b = attrs['transB']
    if 'alpha' in attrs:
        alpha = attrs['alpha']
    if 'beta' in attrs:
        beta = attrs['beta']
    flatten_a = symbol.flatten(inputs[0])
    matmul_op = symbol.linalg_gemm2(A=flatten_a, B=inputs[1], transpose_a=trans_a, transpose_b=trans_b, alpha=alpha)
    gemm_op = symbol.broadcast_add(matmul_op, beta * inputs[2])
    new_attrs = translation_utils._fix_attribute_names(attrs, {'transA': 'transpose_a', 'transB': 'transpose_b'})
    new_attrs = translation_utils._remove_attributes(new_attrs, ['broadcast'])
    return (gemm_op, new_attrs, inputs)