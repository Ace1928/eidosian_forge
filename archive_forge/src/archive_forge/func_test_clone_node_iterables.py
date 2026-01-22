import pytest
from ..base import EngineBase
from ....interfaces import base as nib
from ....interfaces import utility as niu
from ... import engine as pe
def test_clone_node_iterables(tmpdir):
    tmpdir.chdir()

    def addstr(string):
        return '%s + 2' % string
    subject_list = ['sub-001', 'sub-002']
    inputnode = pe.Node(niu.IdentityInterface(fields=['subject']), name='inputnode')
    inputnode.iterables = [('subject', subject_list)]
    node_1 = pe.Node(niu.Function(input_names='string', output_names='string', function=addstr), name='node_1')
    node_2 = node_1.clone('node_2')
    workflow = pe.Workflow(name='iter_clone_wf')
    workflow.connect([(inputnode, node_1, [('subject', 'string')]), (node_1, node_2, [('string', 'string')])])
    workflow.run()