import pytest
from .... import config
from ... import engine as pe
from ....interfaces import base as nib
from ....interfaces.utility import IdentityInterface, Function, Merge
from ....interfaces.base import traits, File
def test_set_join_node_file_input(tmpdir):
    """Test collecting join inputs to a set."""
    tmpdir.chdir()
    open('test.nii', 'w+').close()
    open('test2.nii', 'w+').close()
    wf = pe.Workflow(name='test')
    inputspec = pe.Node(IdentityInterface(fields=['n']), name='inputspec')
    inputspec.iterables = [('n', [tmpdir.join('test.nii').strpath, tmpdir.join('test2.nii').strpath])]
    pre_join1 = pe.Node(IdentityInterface(fields=['n']), name='pre_join1')
    wf.connect(inputspec, 'n', pre_join1, 'n')
    join = pe.JoinNode(PickFirst(), joinsource='inputspec', joinfield='in_files', name='join')
    wf.connect(pre_join1, 'n', join, 'in_files')
    wf.run()