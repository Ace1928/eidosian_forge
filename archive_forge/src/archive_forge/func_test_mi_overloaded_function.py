import typing
import unittest
import onnx
import onnx.parser
import onnx.shape_inference
def test_mi_overloaded_function(self):
    """Test use of functions."""
    model = '\n            <ir_version: 10, opset_import: [ "" : 17, "local" : 1]>\n            agraph (float[N] x) => (y, z)\n            {\n                y = local.cast:to_int32 (x)\n                z = local.cast:to_int64 (x)\n            }\n            <opset_import: [ "" : 17 ], domain: "local", overload: "to_int32">\n            cast (x) => (y)\n            {\n                y = Cast<to=6> (x)\n            }\n            <opset_import: [ "" : 17 ], domain: "local", overload: "to_int64">\n            cast (x) => (y)\n            {\n                y = Cast<to=7> (x)\n            }\n        '
    self._check(model, onnx.TensorProto.INT32, onnx.TensorProto.INT64)