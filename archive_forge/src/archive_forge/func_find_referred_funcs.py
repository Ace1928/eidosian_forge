from __future__ import annotations
import os
import onnx.checker
import onnx.helper
import onnx.shape_inference
from onnx import FunctionProto, ModelProto, NodeProto, TensorProto, ValueInfoProto
def find_referred_funcs(nodes, referred_local_functions):
    new_nodes = []
    for node in nodes:
        match_function = next((f for f in self.model.functions if f.name == node.op_type and f.domain == node.domain), None)
        if match_function and match_function not in referred_local_functions:
            referred_local_functions.append(match_function)
            new_nodes.extend(match_function.node)
    return new_nodes