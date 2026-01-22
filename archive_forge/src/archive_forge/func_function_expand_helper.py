import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import onnx
from onnx.backend.test.case.test_case import TestCase
from onnx.backend.test.case.utils import import_recursive
from onnx.onnx_pb import (
def function_expand_helper(node: NodeProto, function_proto: FunctionProto, op_prefix: str) -> List[NodeProto]:
    io_names_map = {}
    attribute_map = {a.name: a for a in node.attribute}
    for idx in range(len(function_proto.input)):
        io_names_map[function_proto.input[idx]] = node.input[idx] if idx in range(len(node.input)) else ''
    for idx in range(len(function_proto.output)):
        if idx in range(len(node.output)) and node.output[idx] != '':
            io_names_map[function_proto.output[idx]] = node.output[idx]

    def rename_helper(internal_name: str) -> Any:
        if internal_name in io_names_map:
            return io_names_map[internal_name]
        elif internal_name == '':
            return ''
        return op_prefix + internal_name
    new_node_list = [_rename_edges_helper(internal_node, rename_helper, attribute_map, op_prefix) for internal_node in function_proto.node]
    return new_node_list