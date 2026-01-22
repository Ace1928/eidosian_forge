import re
import logging
import numpy as np
from .._export_onnx import MXNetGraph as mx_op
def get_input_dtypes(node, kwargs):
    outputs_lookup = kwargs['outputs_lookup']
    inputs = node['inputs']
    input_dtypes = []
    for ip in inputs:
        input_node_dtype = outputs_lookup[ip[0]][ip[1]].dtype
        input_dtypes.append(input_node_dtype)
    return input_dtypes