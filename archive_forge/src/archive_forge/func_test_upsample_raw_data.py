from __future__ import annotations
import itertools
import unittest
from typing import Any, Sequence
import numpy as np
import pytest
from parameterized import parameterized
import onnx.shape_inference
from onnx import (
from onnx.defs import (
from onnx.helper import (
from onnx.parser import parse_graph
@parameterized.expand(all_versions_for('Upsample'))
def test_upsample_raw_data(self, _, version) -> None:
    if version == 7:
        graph = self._make_graph([('x', TensorProto.INT32, (1, 3, 4, 5))], [make_node('Upsample', ['x'], ['y'], scales=[2.0, 1.1, 2.3, 1.9])], [])
        self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.INT32, (2, 3, 9, 9))], opset_imports=[helper.make_opsetid(ONNX_DOMAIN, version)])
    else:
        graph = self._make_graph([('x', TensorProto.INT32, (2, 4, 3, 5)), ('scales', TensorProto.FLOAT, (4,))], [make_node('Upsample', ['x', 'scales'], ['y'])], [], initializer=[make_tensor('scales', TensorProto.FLOAT, (4,), vals=np.array([1.0, 1.1, 1.3, 1.9], dtype='<f4').tobytes(), raw=True)])

        def call_inference():
            self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.INT32, (2, 4, 3, 9))], opset_imports=[helper.make_opsetid(ONNX_DOMAIN, version)])
        if version == 9:
            call_inference()
        else:
            with self.assertRaises(onnx.checker.ValidationError) as cm:
                call_inference()
            exception = cm.exception
            assert 'Upsample is deprecated' in str(exception)