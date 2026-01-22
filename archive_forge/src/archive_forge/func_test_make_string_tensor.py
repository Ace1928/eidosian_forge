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
def test_make_string_tensor(self) -> None:
    string_list = [s.encode('utf-8') for s in ['Amy', 'Billy', 'Cindy', 'David']]
    tensor = helper.make_tensor(name='test', data_type=TensorProto.STRING, dims=(2, 2), vals=string_list, raw=False)
    self.assertEqual(string_list, list(tensor.string_data))