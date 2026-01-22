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
def test_attr_repeated_tensor_proto(self) -> None:
    tensors = [helper.make_tensor(name='a', data_type=TensorProto.FLOAT, dims=(1,), vals=np.ones(1)), helper.make_tensor(name='b', data_type=TensorProto.FLOAT, dims=(1,), vals=np.ones(1))]
    attr = helper.make_attribute('tensors', tensors)
    self.assertEqual(attr.name, 'tensors')
    self.assertEqual(list(attr.tensors), tensors)
    checker.check_attribute(attr)