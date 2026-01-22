import typing
import unittest
import onnx
import onnx.parser
import onnx.shape_inference
def test_mi_function_attr(self):
    """Test use of functions with attribute parameters."""
    model = '\n            <\n                ir_version: 7,\n                opset_import: [ "" : 17, "local" : 1]\n            >\n            agraph (float[N] x) => (y)\n            {\n                y = local.cast<target=6>(x)\n            }\n            <\n                opset_import: [ "" : 17 ],\n                domain: "local"\n            >\n            cast<target>(x) => (y)\n            {\n                y = Cast<to:int = @target> (x)\n            }\n        '
    self._check(model, onnx.TensorProto.INT32)