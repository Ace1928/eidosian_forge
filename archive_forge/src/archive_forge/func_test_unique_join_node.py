import pytest
from .... import config
from ... import engine as pe
from ....interfaces import base as nib
from ....interfaces.utility import IdentityInterface, Function, Merge
from ....interfaces.base import traits, File
def test_unique_join_node(tmpdir):
    """Test join with the ``unique`` flag set to True."""
    global _sum_operands
    _sum_operands = []
    tmpdir.chdir()
    wf = pe.Workflow(name='test')
    inputspec = pe.Node(IdentityInterface(fields=['n']), name='inputspec')
    inputspec.iterables = [('n', [3, 1, 2, 1, 3])]
    pre_join1 = pe.Node(IncrementInterface(), name='pre_join1')
    wf.connect(inputspec, 'n', pre_join1, 'input1')
    join = pe.JoinNode(SumInterface(), joinsource='inputspec', joinfield='input1', unique=True, name='join')
    wf.connect(pre_join1, 'output1', join, 'input1')
    wf.run()
    assert _sum_operands[0] == [4, 2, 3], 'The unique join output value is incorrect: %s.' % _sum_operands[0]