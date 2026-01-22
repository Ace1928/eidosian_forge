import pytest
from .... import config
from ... import engine as pe
from ....interfaces import base as nib
from ....interfaces.utility import IdentityInterface, Function, Merge
from ....interfaces.base import traits, File
def test_multifield_join_node(tmpdir):
    """Test join on several fields."""
    global _products
    _products = []
    tmpdir.chdir()
    wf = pe.Workflow(name='test')
    inputspec = pe.Node(IdentityInterface(fields=['m', 'n']), name='inputspec')
    inputspec.iterables = [('m', [1, 2]), ('n', [3, 4])]
    inc1 = pe.Node(IncrementInterface(), name='inc1')
    wf.connect(inputspec, 'm', inc1, 'input1')
    inc2 = pe.Node(IncrementInterface(), name='inc2')
    wf.connect(inputspec, 'n', inc2, 'input1')
    join = pe.JoinNode(IdentityInterface(fields=['vector1', 'vector2']), joinsource='inputspec', name='join')
    wf.connect(inc1, 'output1', join, 'vector1')
    wf.connect(inc2, 'output1', join, 'vector2')
    prod = pe.MapNode(ProductInterface(), name='prod', iterfield=['input1', 'input2'])
    wf.connect(join, 'vector1', prod, 'input1')
    wf.connect(join, 'vector2', prod, 'input2')
    result = wf.run()
    assert len(result.nodes()) == 10, 'The number of expanded nodes is incorrect.'
    assert set(_products) == set([8, 10, 12, 15]), 'The post-join products is incorrect: %s.' % _products