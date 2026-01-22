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
def test_make_optional_value_info(self) -> None:
    tensor_type_proto = helper.make_tensor_type_proto(elem_type=2, shape=[5])
    tensor_val_into = helper.make_value_info(name='test', type_proto=tensor_type_proto)
    optional_type_proto = helper.make_optional_type_proto(tensor_type_proto)
    optional_val_info = helper.make_value_info(name='test', type_proto=optional_type_proto)
    self.assertEqual(optional_val_info.name, 'test')
    self.assertTrue(optional_val_info.type.optional_type)
    self.assertEqual(optional_val_info.type.optional_type.elem_type, tensor_val_into.type)
    sequence_type_proto = helper.make_sequence_type_proto(tensor_type_proto)
    optional_type_proto = helper.make_optional_type_proto(sequence_type_proto)
    optional_val_info = helper.make_value_info(name='test', type_proto=optional_type_proto)
    self.assertEqual(optional_val_info.name, 'test')
    self.assertTrue(optional_val_info.type.optional_type)
    sequence_value_info = helper.make_value_info(name='test', type_proto=tensor_type_proto)
    self.assertEqual(optional_val_info.type.optional_type.elem_type.sequence_type.elem_type, sequence_value_info.type)