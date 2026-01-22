import unittest
from typing import Callable, List, Optional, Sequence, Tuple
import numpy as np
from onnx import (
def test_merge_drop_unnecessary_initializers_and_value_info(self) -> None:
    """Tests automatic removal of initializers when merging graphs"""
    ops = [helper.make_opsetid('', 10)]
    g = GraphProto()
    g.input.extend([helper.make_tensor_value_info('x', TensorProto.FLOAT, [])])
    g.output.extend([helper.make_tensor_value_info('y', TensorProto.FLOAT, [])])
    g.node.extend([helper.make_node('Identity', inputs=['x'], outputs=['y'])])
    g1 = GraphProto()
    g1.CopyFrom(g)
    g1.name = 'g1'
    m1 = helper.make_model(g1, producer_name='test', opset_imports=ops)
    checker.check_model(m1)
    g2 = GraphProto()
    g2.CopyFrom(g)
    g2.name = 'g2'
    g2.initializer.extend([helper.make_tensor(name='x', data_type=TensorProto.FLOAT, dims=(), vals=[0])])
    m2 = helper.make_model(g2, producer_name='test', opset_imports=ops)
    checker.check_model(m2)
    g3 = GraphProto()
    g3.CopyFrom(g)
    g3.name = 'g3'
    g3.sparse_initializer.extend([_make_sparse_tensor('x')])
    m3 = helper.make_model(g3, producer_name='test', opset_imports=ops)
    checker.check_model(m3)
    g4 = GraphProto()
    g4.CopyFrom(g)
    g4.name = 'g3'
    g4.value_info.extend([helper.make_tensor_value_info('x', TensorProto.FLOAT, [])])
    m4 = helper.make_model(g4, producer_name='test', opset_imports=ops)
    checker.check_model(m4)
    out_m1 = compose.merge_models(m1, m2, prefix1='m1/', io_map=[('y', 'x')])
    self.assertEqual(0, len(out_m1.graph.initializer))
    out_m2 = compose.merge_models(m1, m3, prefix1='m1/', io_map=[('y', 'x')])
    self.assertEqual(0, len(out_m2.graph.initializer))
    out_m3 = compose.merge_models(m1, m4, prefix1='m1/', io_map=[('y', 'x')])
    self.assertEqual(0, len(out_m3.graph.value_info))