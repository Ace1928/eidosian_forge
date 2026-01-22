import itertools
import random
import struct
import unittest
from typing import Any, List, Tuple
import numpy as np
import parameterized
import pytest
import version_utils
from onnx import (
from onnx.reference.op_run import to_array_extended
def test_make_sparse_tensor(self) -> None:
    values = [1.1, 2.2, 3.3, 4.4, 5.5]
    values_tensor = helper.make_tensor(name='test', data_type=TensorProto.FLOAT, dims=(5,), vals=values)
    indices = [1, 3, 5, 7, 9]
    indices_tensor = helper.make_tensor(name='test_indices', data_type=TensorProto.INT64, dims=(5,), vals=indices)
    dense_shape = [10]
    sparse = helper.make_sparse_tensor(values_tensor, indices_tensor, dense_shape)
    self.assertEqual(sparse.values, values_tensor)
    self.assertEqual(sparse.indices, indices_tensor)
    self.assertEqual(sparse.dims, dense_shape)