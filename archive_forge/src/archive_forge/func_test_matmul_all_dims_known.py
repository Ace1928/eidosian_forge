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
@parameterized.expand(all_versions_for('MatMul'))
def test_matmul_all_dims_known(self, _, version) -> None:
    self._make_matmul_test_all_dims_known(version, (2,), (2,))
    self._make_matmul_test_all_dims_known(version, (4, 2), (2, 4))
    self._make_matmul_test_all_dims_known(version, (5, 2), (2, 4))
    self._make_matmul_test_all_dims_known(version, (5, 2), (2, 1))
    self._make_matmul_test_all_dims_known(version, (1, 2), (2, 3))
    self._make_matmul_test_all_dims_known(version, (2,), (2, 3))
    self._make_matmul_test_all_dims_known(version, (4, 2), (2,))
    self._make_matmul_test_all_dims_known(version, (1, 4, 2), (3, 2, 3))
    self._make_matmul_test_all_dims_known(version, (3, 4, 2), (3, 2, 3))
    self._make_matmul_test_all_dims_known(version, (5, 1, 4, 2), (1, 3, 2, 3))
    self._make_matmul_test_all_dims_known(version, (4, 2), (3, 2, 3))