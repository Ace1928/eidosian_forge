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
def mk_model(opset_versions: List[Tuple[str, int]]) -> ModelProto:
    graph = helper.make_graph([], 'my graph', [], [])
    return helper.make_model_gen_version(graph, opset_imports=[helper.make_opsetid(*pair) for pair in opset_versions])