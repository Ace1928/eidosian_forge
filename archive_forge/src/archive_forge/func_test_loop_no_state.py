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
def test_loop_no_state(self) -> None:
    input_value_infos = [make_tensor_value_info('iter_num_in', TensorProto.INT64, (1,)), make_tensor_value_info('cond_in', TensorProto.UNDEFINED, None)]
    output_value_infos = [make_tensor_value_info('cond_out', TensorProto.UNDEFINED, None), make_tensor_value_info('output', TensorProto.FLOAT, (3,))]
    subgraph = helper.make_graph([make_node('Identity', ['cond_in'], ['cond_out']), make_node('Identity', ['outer_scope_input'], ['output'])], 'subgraph', input_value_infos, output_value_infos)
    graph = self._make_graph([('max_trip_count', TensorProto.INT64, (1,)), ('cond_orig', TensorProto.FLOAT, (1,)), ('outer_scope_input', TensorProto.FLOAT, (3,))], [make_node('Loop', ['max_trip_count', 'cond_orig'], ['loop_output'], body=subgraph)], [])
    self._assert_inferred(graph, [make_tensor_value_info('loop_output', TensorProto.FLOAT, (None, 3))])