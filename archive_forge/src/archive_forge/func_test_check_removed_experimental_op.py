import os
import tempfile
import unittest
from typing import Sequence
import numpy as np
import onnx.defs
import onnx.parser
from onnx import (
def test_check_removed_experimental_op(self) -> None:
    node = helper.make_node('ConstantFill', [], ['Y'], name='test', shape=[1, 2])
    checker.check_node(node)