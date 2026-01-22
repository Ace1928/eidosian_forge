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
def test_matmulinteger(self) -> None:
    self._make_matmulinteger_test((2,), (2,))
    self._make_matmulinteger_test((1, 2), (2, 3))
    self._make_matmulinteger_test((2,), (2, 3))
    self._make_matmulinteger_test((4, 2), (2,))
    self._make_matmulinteger_test((5, 1, 4, 2), (1, 3, 2, 3))
    self._make_matmulinteger_test((4, 2), (3, 2, 3))