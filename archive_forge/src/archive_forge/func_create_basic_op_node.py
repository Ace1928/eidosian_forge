import re
import logging
import numpy as np
from .._export_onnx import MXNetGraph as mx_op
def create_basic_op_node(op_name, node, kwargs):
    """Helper function to create a basic operator
    node that doesn't contain op specific attrs"""
    name, input_nodes, _ = get_inputs(node, kwargs)
    node = onnx.helper.make_node(op_name, input_nodes, [name], name=name)
    return [node]