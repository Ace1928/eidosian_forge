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
def test_attr_repeated_float(self) -> None:
    attr = helper.make_attribute('floats', [1.0, 2.0])
    self.assertEqual(attr.name, 'floats')
    self.assertEqual(list(attr.floats), [1.0, 2.0])
    checker.check_attribute(attr)