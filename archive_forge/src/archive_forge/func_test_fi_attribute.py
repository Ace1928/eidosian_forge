import unittest
from typing import Sequence
from shape_inference_test import TestShapeInferenceHelper
import onnx
import onnx.helper
import onnx.parser
import onnx.shape_inference
from onnx import AttributeProto, TypeProto
def test_fi_attribute(self):
    code = '\n            <opset_import: [ "" : 18 ], domain: "local">\n            CastTo <dtype> (x) => (y) {\n                y = Cast <to : int = @dtype> (x)\n            }\n        '
    dtype_6 = onnx.helper.make_attribute('dtype', 6)
    self._check(code, [float_type_], [dtype_6], [int32_type_])
    dtype_10 = onnx.helper.make_attribute('dtype', 10)
    self._check(code, [float_type_], [dtype_10], [float16_type_])