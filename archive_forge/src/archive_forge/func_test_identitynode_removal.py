import os
from copy import deepcopy
import pytest
from ... import engine as pe
from ....interfaces import base as nib
from ....interfaces import utility as niu
from .... import config
from ..utils import (
def test_identitynode_removal(tmpdir):

    def test_function(arg1, arg2, arg3):
        import numpy as np
        return (np.array(arg1) + arg2 + arg3).tolist()
    wf = pe.Workflow(name='testidentity', base_dir=tmpdir.strpath)
    n1 = pe.Node(niu.IdentityInterface(fields=['a', 'b']), name='src', base_dir=tmpdir.strpath)
    n1.iterables = ('b', [0, 1, 2, 3])
    n1.inputs.a = [0, 1, 2, 3]
    n2 = pe.Node(niu.Select(), name='selector', base_dir=tmpdir.strpath)
    wf.connect(n1, ('a', test_function, 1, -1), n2, 'inlist')
    wf.connect(n1, 'b', n2, 'index')
    n3 = pe.Node(niu.IdentityInterface(fields=['c', 'd']), name='passer', base_dir=tmpdir.strpath)
    n3.inputs.c = [1, 2, 3, 4]
    wf.connect(n2, 'out', n3, 'd')
    n4 = pe.Node(niu.Select(), name='selector2', base_dir=tmpdir.strpath)
    wf.connect(n3, ('c', test_function, 1, -1), n4, 'inlist')
    wf.connect(n3, 'd', n4, 'index')
    fg = wf._create_flat_graph()
    wf._set_needed_outputs(fg)
    eg = pe.generate_expanded_graph(deepcopy(fg))
    assert len(eg.nodes()) == 8