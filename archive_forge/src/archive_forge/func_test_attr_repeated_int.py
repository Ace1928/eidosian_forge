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
def test_attr_repeated_int(self) -> None:
    attr = helper.make_attribute('ints', [1, 2])
    self.assertEqual(attr.name, 'ints')
    self.assertEqual(list(attr.ints), [1, 2])
    checker.check_attribute(attr)