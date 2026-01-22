import typing
import unittest
import onnx
import onnx.parser
import onnx.shape_inference
def test_mi_function_subgraph_attr(self):
    """Test use of function attributes within subgraphs."""
    model = '\n            <\n                ir_version: 7,\n                opset_import: [ "" : 17, "local" : 1]\n            >\n            agraph (float[N] x, bool flag) => (y)\n            {\n                y = local.cast<target=6>(x, flag)\n            }\n            <\n                opset_import: [ "" : 17 ],\n                domain: "local"\n            >\n            cast<target>(x, flag) => (y)\n            {\n                y = If (flag) <\n                    then_branch = g1 () => (z_then) { z_then = Cast<to:int = @target> (x) },\n                    else_branch = g2 () => (z_else) { z_else = Cast<to:int = @target> (x) }\n                    >\n            }\n        '
    self._check(model, onnx.TensorProto.INT32)