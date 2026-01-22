import copy
import os
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Union
import onnx
from onnx import ModelProto
from ..utils import logging
from .transformations_utils import (
def cast_slice_nodes_inputs_to_int32(model: ModelProto) -> ModelProto:
    """
    Convert node inputs of `Slice` nodes from int64 to int32, casting the out of range values.

    The constant node inputs are stored in `model.graph.node`, and the sole way to check which node
    they are consumed by is to iterate over nodes and check `node.input` for a match.

    Note that constant inputs to nodes as `Squeeze`, `Unsqueeze` can not be converted to int32, as the
    these operators explicitely expect int64 inputs according to ONNX specifications:
    https://github.com/onnx/onnx/blob/main/docs/Operators.md
    """
    map_input_node = {}
    map_node_inputs = {}
    for node in model.graph.node:
        for input_name in node.input:
            map_input_node[input_name] = {'op_type': node.op_type, 'node_name': node.name}
        map_node_inputs[node.name] = node.input
    for node in model.graph.node:
        if node.op_type == 'Constant' and node.attribute[0].t.data_type == 7 and (f'{node.name}_output_0' in map_input_node) and (map_input_node[node.name + '_output_0']['op_type'] == 'Slice'):
            logger.debug(f'Converting {node.name} to int32')
            cast = all(('Constant' in inp for inp in map_node_inputs[map_input_node[node.name + '_output_0']['node_name']][1:]))
            cast_int64_tensorproto_to_int32(node.attribute[0].t, cast=cast)
    return model