import pytest
from .... import config
from ... import engine as pe
from ....interfaces import base as nib
from ....interfaces.utility import IdentityInterface, Function, Merge
from ....interfaces.base import traits, File
def test_multiple_join_nodes(tmpdir):
    """Test two join nodes, one downstream of the other."""
    global _products
    _products = []
    tmpdir.chdir()
    wf = pe.Workflow(name='test')
    inputspec = pe.Node(IdentityInterface(fields=['n']), name='inputspec')
    inputspec.iterables = [('n', [1, 2, 3])]
    pre_join1 = pe.Node(IncrementInterface(), name='pre_join1')
    wf.connect(inputspec, 'n', pre_join1, 'input1')
    join1 = pe.JoinNode(IdentityInterface(fields=['vector']), joinsource='inputspec', joinfield='vector', name='join1')
    wf.connect(pre_join1, 'output1', join1, 'vector')
    post_join1 = pe.Node(SumInterface(), name='post_join1')
    wf.connect(join1, 'vector', post_join1, 'input1')
    join2 = pe.JoinNode(IdentityInterface(fields=['vector', 'scalar']), joinsource='inputspec', joinfield='vector', name='join2')
    wf.connect(pre_join1, 'output1', join2, 'vector')
    wf.connect(post_join1, 'output1', join2, 'scalar')
    post_join2 = pe.Node(SumInterface(), name='post_join2')
    wf.connect(join2, 'vector', post_join2, 'input1')
    post_join3 = pe.Node(ProductInterface(), name='post_join3')
    wf.connect(post_join2, 'output1', post_join3, 'input1')
    wf.connect(join2, 'scalar', post_join3, 'input2')
    result = wf.run()
    assert len(result.nodes()) == 8, 'The number of expanded nodes is incorrect.'
    assert _products == [81], 'The post-join product is incorrect'