import unittest
from parameterized import parameterized
import onnx
from onnx import GraphProto, OperatorSetIdProto, checker
def test_parse_graph(self) -> None:
    input = '\n           agraph (float[N, 128] X, float[128,10] W, float[10] B) => (float[N] C)\n           {\n              T = MatMul(X, W)\n              S = Add(T, B)\n              C = Softmax(S)\n           }\n           '
    graph = onnx.parser.parse_graph(input)
    self.check_graph(graph)