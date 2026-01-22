import pytest
from .... import config
from ... import engine as pe
from ....interfaces import base as nib
from ....interfaces.utility import IdentityInterface, Function, Merge
from ....interfaces.base import traits, File
def nested_wf(i, name='smallwf'):
    inputspec = pe.Node(IdentityInterface(fields=['n']), name='inputspec')
    inputspec.iterables = [('n', i)]
    pre_join = pe.Node(IncrementInterface(), name='pre_join')
    join = pe.JoinNode(IdentityInterface(fields=['n']), joinsource='inputspec', joinfield='n', name='join')
    wf = pe.Workflow(name='wf_%d' % i[0])
    wf.connect(inputspec, 'n', pre_join, 'input1')
    wf.connect(pre_join, 'output1', join, 'n')
    return wf