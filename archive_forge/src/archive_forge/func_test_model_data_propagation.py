from __future__ import annotations
import unittest
from shape_inference_test import TestShapeInferenceHelper
import onnx.parser
from onnx import TensorProto
from onnx.helper import make_node, make_tensor, make_tensor_value_info
def test_model_data_propagation(self) -> None:
    """Infer the shape of z by propagating the value of xshape."""
    model = onnx.parser.parse_model('\n            <ir_version: 7, opset_import: [ "" : 18]>\n            agraph (float[4, 1, 16] x, float[1, 8, 16] y) => () {\n                xshape = Shape (x)\n                z = Expand (y, xshape)\n            }\n        ')
    self._assert_inferred(model, [make_tensor_value_info('xshape', TensorProto.INT64, (3,)), make_tensor_value_info('z', TensorProto.FLOAT, (4, 8, 16))], data_prop=True)