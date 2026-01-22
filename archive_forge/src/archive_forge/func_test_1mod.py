from copy import deepcopy
from glob import glob
import os
import pytest
from ... import engine as pe
from .test_base import EngineTestInterface
import networkx
@pytest.mark.parametrize('iterables, expected', [({'1': None}, (1, 0)), ({'1': dict(input1=lambda: [1, 2], input2=lambda: [1, 2])}, (4, 0))])
def test_1mod(iterables, expected):
    pipe = pe.Workflow(name='pipe')
    mod1 = pe.Node(interface=EngineTestInterface(), name='mod1')
    setattr(mod1, 'iterables', iterables['1'])
    pipe.add_nodes([mod1])
    pipe._flatgraph = pipe._create_flat_graph()
    pipe._execgraph = pe.generate_expanded_graph(deepcopy(pipe._flatgraph))
    assert len(pipe._execgraph.nodes()) == expected[0]
    assert len(pipe._execgraph.edges()) == expected[1]