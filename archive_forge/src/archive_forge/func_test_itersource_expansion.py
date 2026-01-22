from copy import deepcopy
from glob import glob
import os
import pytest
from ... import engine as pe
from .test_base import EngineTestInterface
import networkx
def test_itersource_expansion():
    wf1 = pe.Workflow(name='test')
    node1 = pe.Node(EngineTestInterface(), name='node1')
    node1.iterables = ('input1', [1, 2])
    node2 = pe.Node(EngineTestInterface(), name='node2')
    wf1.connect(node1, 'output1', node2, 'input1')
    node3 = pe.Node(EngineTestInterface(), name='node3')
    node3.itersource = ('node1', 'input1')
    node3.iterables = [('input1', {1: [3, 4], 2: [5, 6, 7]})]
    wf1.connect(node2, 'output1', node3, 'input1')
    node4 = pe.Node(EngineTestInterface(), name='node4')
    wf1.connect(node3, 'output1', node4, 'input1')
    wf3 = pe.Workflow(name='group')
    for i in [0, 1, 2]:
        wf3.add_nodes([wf1.clone(name='test%d' % i)])
    wf3._flatgraph = wf3._create_flat_graph()
    assert len(pe.generate_expanded_graph(wf3._flatgraph).nodes()) == 42