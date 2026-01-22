import pytest
from .... import config
from ... import engine as pe
from ....interfaces import base as nib
from ....interfaces.utility import IdentityInterface, Function, Merge
from ....interfaces.base import traits, File
@pytest.mark.parametrize('needed_outputs', ['true', 'false'])
def test_join_expansion(tmpdir, needed_outputs):
    global _sums
    global _sum_operands
    global _products
    tmpdir.chdir()
    _products = []
    _sum_operands = []
    _sums = []
    prev_state = config.get('execution', 'remove_unnecessary_outputs')
    config.set('execution', 'remove_unnecessary_outputs', needed_outputs)
    wf = pe.Workflow(name='test')
    inputspec = pe.Node(IdentityInterface(fields=['n']), name='inputspec')
    inputspec.iterables = [('n', [1, 2])]
    pre_join1 = pe.Node(IncrementInterface(), name='pre_join1')
    pre_join2 = pe.Node(IncrementInterface(), name='pre_join2')
    join = pe.JoinNode(SumInterface(), joinsource='inputspec', joinfield='input1', name='join')
    post_join1 = pe.Node(IncrementInterface(), name='post_join1')
    post_join2 = pe.Node(ProductInterface(), name='post_join2')
    wf.connect([(inputspec, pre_join1, [('n', 'input1')]), (pre_join1, pre_join2, [('output1', 'input1')]), (pre_join1, post_join2, [('output1', 'input2')]), (pre_join2, join, [('output1', 'input1')]), (join, post_join1, [('output1', 'input1')]), (join, post_join2, [('output1', 'input1')])])
    result = wf.run()
    joins = [node for node in result.nodes() if node.name == 'join']
    assert len(joins) == 1, 'The number of join result nodes is incorrect.'
    assert len(result.nodes()) == 8, 'The number of expanded nodes is incorrect.'
    assert len(_sums) == 1, 'The number of join outputs is incorrect'
    assert _sums[0] == 7, 'The join Sum output value is incorrect: %s.' % _sums[0]
    assert _sum_operands[0] == [3, 4], 'The join Sum input is incorrect: %s.' % _sum_operands[0]
    assert len(_products) == 2, 'The number of iterated post-join outputs is incorrect'
    config.set('execution', 'remove_unnecessary_outputs', prev_state)