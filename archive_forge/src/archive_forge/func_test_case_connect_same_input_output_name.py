import unittest
from typing import Callable, List, Optional, Sequence, Tuple
import numpy as np
from onnx import (
def test_case_connect_same_input_output_name(self) -> None:
    """Tests a scenario where we merge two models, where the inputs/outputs connected
        are named exactly the same
        """
    m1_def = '\n            <\n                ir_version: 7,\n                opset_import: [ "": 10]\n            >\n            agraph (float[N, M] A) => (float[N, M] B)\n            {\n                B = Add(A, A)\n            }\n            '
    m2_def = '\n            <\n                ir_version: 7,\n                opset_import: [ "": 10]\n            >\n            agraph (float[N, M] B) => (float[N, M] C)\n            {\n                C = Add(B, B)\n            }\n            '
    io_map = [('B', 'B')]

    def check_expectations(g1: GraphProto, g2: GraphProto, g3: GraphProto) -> None:
        del g1, g2
        self.assertEqual(['A'], [elem.name for elem in g3.input])
        self.assertEqual(['C'], [elem.name for elem in g3.output])
    self._test_merge_models(m1_def, m2_def, io_map, check_expectations)