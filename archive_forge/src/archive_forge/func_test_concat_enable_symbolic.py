import unittest
from typing import List, Optional
import onnx.shape_inference
from onnx import ModelProto, TensorProto, TensorShapeProto, ValueInfoProto, helper
from onnx.helper import make_model, make_tensor_value_info
def test_concat_enable_symbolic(self) -> None:
    concat = helper.make_node('Concat', inputs=['A', 'B'], outputs=['C'], name='Concat', axis=1)
    cast = onnx.helper.make_node('Cast', inputs=['C'], outputs=['output'], to=TensorProto.FLOAT)
    graph_def = helper.make_graph(name='test_graph', nodes=[concat, cast], inputs=[helper.make_tensor_value_info('A', TensorProto.FLOAT, [2, 'A']), helper.make_tensor_value_info('B', TensorProto.FLOAT, [2, 3])], outputs=[helper.make_tensor_value_info('output', TensorProto.FLOAT, [2, None])])
    onnx_model = make_model(graph_def)
    inferred_model = onnx.shape_inference.infer_shapes(onnx_model, strict_mode=True)
    self._assert_valueinfo_shape(inferred_model, [make_tensor_value_info('C', TensorProto.FLOAT, (2, -1))])
    assert self._get_shape_from_name(inferred_model, 'C') == self._get_shape_from_name(inferred_model, 'output')