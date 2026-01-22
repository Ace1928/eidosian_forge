from copy import deepcopy
from glob import glob
import os
import pytest
from ... import engine as pe
from .test_base import EngineTestInterface
import networkx
@pytest.mark.parametrize('iterables, expected', [({'1': {}, '2': dict(input1=lambda: [1, 2])}, (3, 2)), ({'1': dict(input1=lambda: [1, 2]), '2': {}}, (4, 2)), ({'1': dict(input1=lambda: [1, 2]), '2': dict(input1=lambda: [1, 2])}, (6, 4))])
def test_2mods(iterables, expected):
    pipe = pe.Workflow(name='pipe')
    mod1 = pe.Node(interface=EngineTestInterface(), name='mod1')
    mod2 = pe.Node(interface=EngineTestInterface(), name='mod2')
    for nr in ['1', '2']:
        setattr(eval('mod' + nr), 'iterables', iterables[nr])
    pipe.connect([(mod1, mod2, [('output1', 'input2')])])
    pipe._flatgraph = pipe._create_flat_graph()
    pipe._execgraph = pe.generate_expanded_graph(deepcopy(pipe._flatgraph))
    assert len(pipe._execgraph.nodes()) == expected[0]
    assert len(pipe._execgraph.edges()) == expected[1]