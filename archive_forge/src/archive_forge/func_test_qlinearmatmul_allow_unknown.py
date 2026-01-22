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
def test_qlinearmatmul_allow_unknown(self) -> None:
    self._make_qlinearmatmul_test_allow_unknown((None,), (None,), ())
    self._make_qlinearmatmul_test_allow_unknown((3,), (None,), ())
    self._make_qlinearmatmul_test_allow_unknown((2,), (2, 'a'), ('a',))
    self._make_qlinearmatmul_test_allow_unknown((4, 2), (2, 'a'), (4, 'a'))
    self._make_qlinearmatmul_test_allow_unknown((4, None), (2, 'a'), (4, 'a'))
    self._make_qlinearmatmul_test_allow_unknown((4, None), (None, 'a'), (4, 'a'))
    self._make_qlinearmatmul_test_allow_unknown((1, 4, 2), ('a', 2, 5), ('a', 4, 5))
    self._make_qlinearmatmul_test_allow_unknown((1, 3, 4, 2), ('a', 2, 5), (1, 3, 4, 5))
    self._make_qlinearmatmul_test_allow_unknown(None, ('a', 2, 5), None)
    self._make_qlinearmatmul_test_allow_unknown(None, None, None)