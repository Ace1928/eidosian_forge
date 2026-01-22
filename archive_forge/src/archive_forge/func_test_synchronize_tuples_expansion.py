from copy import deepcopy
from glob import glob
import os
import pytest
from ... import engine as pe
from .test_base import EngineTestInterface
import networkx
def test_synchronize_tuples_expansion():
    wf1 = pe.Workflow(name='test')
    node1 = pe.Node(EngineTestInterface(), name='node1')
    node2 = pe.Node(EngineTestInterface(), name='node2')
    node1.iterables = [('input1', 'input2'), [(1, 3), (2, 4), (None, 5)]]
    node1.synchronize = True
    wf1.connect(node1, 'output1', node2, 'input2')
    wf3 = pe.Workflow(name='group')
    for i in [0, 1, 2]:
        wf3.add_nodes([wf1.clone(name='test%d' % i)])
    wf3._flatgraph = wf3._create_flat_graph()
    assert len(pe.generate_expanded_graph(wf3._flatgraph).nodes()) == 18