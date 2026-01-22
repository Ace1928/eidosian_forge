import unittest
from parameterized import parameterized
import onnx
from onnx import GraphProto, OperatorSetIdProto, checker
def test_parse_model(self) -> None:
    input = '\n           <\n             ir_version: 7,\n             opset_import: [ "" : 10, "com.microsoft": 1]\n           >\n           agraph (float[N, 128] X, float[128,10] W, float[10] B) => (float[N] C)\n           {\n              T = MatMul(X, W)\n              S = Add(T, B)\n              C = Softmax(S)\n           }\n           '
    model = onnx.parser.parse_model(input)
    self.assertEqual(model.ir_version, 7)
    self.assertEqual(len(model.opset_import), 2)
    self.check_graph(model.graph)