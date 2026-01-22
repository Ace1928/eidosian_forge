import unittest
from typing import List, Optional
import onnx.shape_inference
from onnx import ModelProto, TensorProto, TensorShapeProto, ValueInfoProto, helper
from onnx.helper import make_model, make_tensor_value_info
def test_duplicate_symbolic_shape(self) -> None:
    concat1 = helper.make_node('Concat', inputs=['A', 'B'], outputs=['C'], name='Concat', axis=1)
    concat2 = helper.make_node('Concat', inputs=['C', 'D'], outputs=['E'], name='Concat', axis=1)
    cast = onnx.helper.make_node('Cast', inputs=['E'], outputs=['output'], to=TensorProto.FLOAT)
    graph_def = helper.make_graph(name='test_graph', nodes=[concat1, concat2, cast], inputs=[helper.make_tensor_value_info('A', TensorProto.FLOAT, [2, 'unk__0']), helper.make_tensor_value_info('B', TensorProto.FLOAT, [2, 3]), helper.make_tensor_value_info('D', TensorProto.FLOAT, [2, 'unk__1'])], outputs=[helper.make_tensor_value_info('output', TensorProto.FLOAT, [2, 'unk__0'])])
    onnx_model = make_model(graph_def)
    original_count = self._count_unique_dim_param_number(onnx_model)
    inferred_model = onnx.shape_inference.infer_shapes(onnx_model, strict_mode=True)
    inferred_count = self._count_unique_dim_param_number(inferred_model)
    assert inferred_count == original_count + 2, f'{inferred_model}{onnx_model}'