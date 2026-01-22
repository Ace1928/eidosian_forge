import unittest
from typing import Callable, List, Optional, Sequence, Tuple
import numpy as np
from onnx import (
def test_case_drop_inputs_outputs(self) -> None:
    """Tests a scenario where we merge two models, not including some of the inputs/outputs"""
    m1_def = '\n            <\n                ir_version: 7,\n                opset_import: [ "": 10]\n            >\n            agraph (float[N] A0, float[N] B0) => (float[N] A1, float[N] B1)\n            {\n                A1 = Add(A0, A0)\n                B1 = Sub(B0, B0)\n            }\n            '
    m2_def = '\n            <\n                ir_version: 7,\n                opset_import: [ "": 10]\n            >\n            agraph (float[N] A2, float[N] B2) => (float[N] A3, float[N] B3)\n            {\n                A3 = Add(A2, A2)\n                B3 = Sub(B2, B2)\n            }\n            '
    io_map = [('A1', 'B2')]

    def check_expectations(g1: GraphProto, g2: GraphProto, g3: GraphProto) -> None:
        del g1, g2
        self.assertEqual(['A0'], [elem.name for elem in g3.input])
        self.assertEqual(['B3'], [elem.name for elem in g3.output])
        self.assertEqual(['Add', 'Sub'], [elem.op_type for elem in g3.node])
    inputs = ['A0']
    outputs = ['B3']
    self._test_merge_models(m1_def, m2_def, io_map, check_expectations, inputs=inputs, outputs=outputs)