import typing
import unittest
import onnx
import onnx.parser
import onnx.shape_inference
def test_mi_basic(self):
    """Test that model inference infers model output type."""
    model = '\n            <\n                ir_version: 7,\n                opset_import: [ "" : 17]\n            >\n            agraph (float[N] x) => (y)\n            {\n                y = Cast<to=6> (x)\n            }\n        '
    self._check(model, onnx.TensorProto.INT32)