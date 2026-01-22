import unittest
from typing import Callable, List, Optional, Sequence, Tuple
import numpy as np
from onnx import (
def test_add_prefix_attribute_subgraph(self) -> None:
    """Tests prefixing attribute's subgraph. Relevant subgraph should be renamed as well"""
    C = helper.make_tensor_value_info('C', TensorProto.BOOL, [1])
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [None, 1])
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [None, 1])
    Z = helper.make_tensor_value_info('Z', TensorProto.FLOAT, [None, 1])
    Out = helper.make_tensor_value_info('Out', TensorProto.FLOAT, [None, 1])
    XY = helper.make_node('Mul', inputs=['X', 'Y'], outputs=['XY'])
    add = helper.make_node('Add', inputs=['XY', 'Z'], outputs=['Out'])
    sub = helper.make_node('Sub', inputs=['XY', 'Z'], outputs=['Out'])
    cond = helper.make_node('If', inputs=['C'], outputs=['Out'], then_branch=helper.make_graph(nodes=[add], name='then', inputs=[], outputs=[Out]), else_branch=helper.make_graph(nodes=[sub], name='else', inputs=[], outputs=[Out]))
    graph = helper.make_graph(nodes=[XY, cond], name='graph', inputs=[C, X, Y, Z], outputs=[Out])
    prefix = 'prefix.'
    prefixed_graph = compose.add_prefix_graph(graph, prefix)
    checker.check_graph(prefixed_graph)
    for n1, n0 in zip(prefixed_graph.node, graph.node):
        self.assertEqual(_prefixed(prefix, n0.name), n1.name)
        for attribute1, attribute0 in zip(n1.attribute, n0.attribute):
            if attribute1.g:
                for subgraph_n1, subgraph_n0 in zip(attribute1.g.node, attribute0.g.node):
                    for input_n1, input_n0 in zip(subgraph_n1.input, subgraph_n0.input):
                        self.assertEqual(_prefixed(prefix, input_n0), input_n1)
                    for output_n1, output_n0 in zip(subgraph_n1.output, subgraph_n0.output):
                        self.assertEqual(_prefixed(prefix, output_n0), output_n1)