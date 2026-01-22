import sys
import os
import pytest
from nipype.pipeline import engine as pe
from nipype.interfaces import base as nib
@pytest.mark.skipif(sys.version_info >= (3, 8), reason='multiprocessing issues in Python 3.8')
def test_run_multiproc(tmpdir):
    tmpdir.chdir()
    pipe = pe.Workflow(name='pipe')
    mod1 = pe.Node(MultiprocTestInterface(), name='mod1')
    mod2 = pe.MapNode(MultiprocTestInterface(), iterfield=['input1'], name='mod2')
    pipe.connect([(mod1, mod2, [('output1', 'input1')])])
    pipe.base_dir = os.getcwd()
    mod1.inputs.input1 = 1
    pipe.config['execution']['poll_sleep_duration'] = 2
    execgraph = pipe.run(plugin='MultiProc')
    names = [node.fullname for node in execgraph.nodes()]
    node = list(execgraph.nodes())[names.index('pipe.mod1')]
    result = node.get_output('output1')
    assert result == [1, 1]