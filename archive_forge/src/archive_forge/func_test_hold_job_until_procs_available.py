import sys
import os
import pytest
from nipype.pipeline import engine as pe
from nipype.interfaces import base as nib
@pytest.mark.skipif(sys.version_info >= (3, 8), reason='multiprocessing issues in Python 3.8')
def test_hold_job_until_procs_available(tmpdir):
    tmpdir.chdir()
    pipe = pe.Workflow(name='pipe')
    n1 = pe.Node(SingleNodeTestInterface(), name='n1', n_procs=2)
    n2 = pe.Node(SingleNodeTestInterface(), name='n2', n_procs=2)
    n3 = pe.Node(SingleNodeTestInterface(), name='n3', n_procs=2)
    n4 = pe.Node(SingleNodeTestInterface(), name='n4', n_procs=2)
    pipe.connect(n1, 'output1', n2, 'input1')
    pipe.connect(n1, 'output1', n3, 'input1')
    pipe.connect(n2, 'output1', n4, 'input1')
    pipe.connect(n3, 'output1', n4, 'input2')
    n1.inputs.input1 = 4
    max_threads = 2
    pipe.run(plugin='MultiProc', plugin_args={'n_procs': max_threads})