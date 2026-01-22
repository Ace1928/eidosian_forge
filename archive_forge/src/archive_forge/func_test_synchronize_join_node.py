import pytest
from .... import config
from ... import engine as pe
from ....interfaces import base as nib
from ....interfaces.utility import IdentityInterface, Function, Merge
from ....interfaces.base import traits, File
def test_synchronize_join_node(tmpdir):
    """Test join on an input node which has the ``synchronize`` flag set to True."""
    global _products
    _products = []
    tmpdir.chdir()
    wf = pe.Workflow(name='test')
    inputspec = pe.Node(IdentityInterface(fields=['m', 'n']), name='inputspec')
    inputspec.iterables = [('m', [1, 2]), ('n', [3, 4])]
    inputspec.synchronize = True
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
    assert len(result.nodes()) == 6, 'The number of expanded nodes is incorrect.'
    assert _products == [8, 15], 'The post-join products is incorrect: %s.' % _products