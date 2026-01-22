from __future__ import annotations
import os
from typing import Sequence
import onnx
import onnx.onnx_cpp2py_export.shape_inference as C  # noqa: N812
from onnx import AttributeProto, FunctionProto, ModelProto, TypeProto
def infer_node_outputs(schema: onnx.defs.OpSchema, node: onnx.NodeProto, input_types: dict[str, onnx.TypeProto], input_data: dict[str, onnx.TensorProto] | None=None, input_sparse_data: dict[str, onnx.SparseTensorProto] | None=None, opset_imports: list[onnx.OperatorSetIdProto] | None=None, ir_version: int=onnx.IR_VERSION) -> dict[str, onnx.TypeProto]:
    if not schema.has_type_and_shape_inference_function:
        return {}
    if input_data is None:
        input_data = {}
    if input_sparse_data is None:
        input_sparse_data = {}
    if opset_imports is None:
        passed_opset_imports = {}
    else:
        passed_opset_imports = {opset.domain: opset.version for opset in opset_imports}
    passed_input_types = {key: input_types[key].SerializeToString() for key in node.input}
    for key in input_types:
        if key not in passed_input_types:
            passed_input_types[key] = input_types[key].SerializeToString()
    passed_input_data = {key: input_data[key].SerializeToString() for key in node.input if key in input_data}
    passed_sparse_input_data = {key: input_sparse_data[key].SerializeToString() for key in node.input if key in input_sparse_data}
    outputs = schema._infer_node_outputs(node.SerializeToString(), passed_input_types, passed_input_data, passed_sparse_input_data, passed_opset_imports, ir_version)
    return {key: onnx.TypeProto.FromString(out) for key, out in outputs.items()}