import unittest
from typing import Callable, List, Optional, Sequence, Tuple
import numpy as np
from onnx import (
def test_overlapping_function_names(self) -> None:
    """Tests error checking when the name of local function entries overlaps"""
    ops = [helper.make_opsetid('', 10), helper.make_opsetid('local', 10)]

    def _make_function(domain: str, fname: str, inputs: List[str], outputs: List[str], nodes: List[NodeProto]) -> FunctionProto:
        f = FunctionProto()
        f.domain = domain
        f.name = fname
        f.input.extend(inputs)
        f.output.extend(outputs)
        f.node.extend(nodes)
        f.opset_import.extend(ops)
        return f
    ops = [helper.make_opsetid('', 10), helper.make_opsetid('local', 10)]
    g = GraphProto()
    g.input.extend([helper.make_tensor_value_info('x0', TensorProto.FLOAT, []), helper.make_tensor_value_info('x1', TensorProto.FLOAT, [])])
    g.output.extend([helper.make_tensor_value_info('y', TensorProto.FLOAT, [])])
    g.node.extend([helper.make_node('f1', domain='local', inputs=['x0', 'x1'], outputs=['y'])])
    g1 = GraphProto()
    g1.CopyFrom(g)
    g1.name = 'g1'
    m1 = helper.make_model(g1, producer_name='test', opset_imports=ops)
    m1.functions.extend([_make_function('local', 'f1', ['x0', 'x1'], ['y'], [helper.make_node('Add', inputs=['x0', 'x1'], outputs=['y'])])])
    checker.check_model(m1)
    g2 = GraphProto()
    g2.CopyFrom(g)
    g2.name = 'g2'
    m2 = helper.make_model(g2, producer_name='test', opset_imports=ops)
    m2.functions.extend([_make_function('local', 'f1', ['x0', 'x1'], ['y'], [helper.make_node('Mul', inputs=['x0', 'x1'], outputs=['y'])])])
    checker.check_model(m2)
    m = compose.merge_models(m1, m2, io_map=[('y', 'x0'), ('y', 'x1')], prefix1='m1/', prefix2='m2/')
    checker.check_model(m)
    nodes = [n.op_type for n in m.graph.node]
    self.assertEqual(['m1/f1', 'm2/f1'], nodes)
    functions = [f.name for f in m.functions]
    self.assertEqual(['m1/f1', 'm2/f1'], functions)
    g3 = GraphProto()
    g3.CopyFrom(g)
    g3.name = 'g3'
    g3.node[0].op_type = 'f2'
    m3 = helper.make_model(g3, producer_name='test', opset_imports=ops)
    m3.functions.extend([_make_function('local', 'f1', ['x0', 'x1'], ['y'], [helper.make_node('Add', inputs=['x0', 'x1'], outputs=['y0']), helper.make_node('Mul', inputs=['x0', 'x1'], outputs=['y1']), helper.make_node('Add', inputs=['y0', 'y1'], outputs=['y'])]), _make_function('local', 'f2', ['x0', 'x1'], ['y'], [helper.make_node('f1', domain='local', inputs=['x0', 'x1'], outputs=['y0']), helper.make_node('Mul', inputs=['x0', 'x1'], outputs=['y1']), helper.make_node('Add', inputs=['y0', 'y1'], outputs=['y'])])])
    checker.check_model(m3)
    m = compose.merge_models(m1, m3, io_map=[('y', 'x0'), ('y', 'x1')], prefix1='m1/', prefix2='m3/')
    checker.check_model(m)
    nodes = [n.op_type for n in m.graph.node]
    self.assertEqual(['m1/f1', 'm3/f2'], nodes)
    functions = [f.name for f in m.functions]
    self.assertEqual(['m1/f1', 'm3/f1', 'm3/f2'], functions)
    self.assertEqual(['Add'], [n.op_type for n in m.functions[0].node])
    self.assertEqual(['Add', 'Mul', 'Add'], [n.op_type for n in m.functions[1].node])
    self.assertEqual(['m3/f1', 'Mul', 'Add'], [n.op_type for n in m.functions[2].node])