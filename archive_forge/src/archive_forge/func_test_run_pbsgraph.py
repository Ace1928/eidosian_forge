from shutil import which
import nipype.interfaces.base as nib
import pytest
import nipype.pipeline.engine as pe
@pytest.mark.skipif(which('qsub') is None, reason='PBS not installed')
@pytest.mark.timeout(60)
def test_run_pbsgraph(tmp_path):
    pipe = pe.Workflow(name='pipe', base_dir=str(tmp_path))
    mod1 = pe.Node(interface=PbsTestInterface(), name='mod1')
    mod2 = pe.MapNode(interface=PbsTestInterface(), iterfield=['input1'], name='mod2')
    pipe.connect([(mod1, mod2, [('output1', 'input1')])])
    mod1.inputs.input1 = 1
    execgraph = pipe.run(plugin='PBSGraph')
    names = ['.'.join((node._hierarchy, node.name)) for node in execgraph.nodes()]
    node = list(execgraph.nodes())[names.index('pipe.mod1')]
    result = node.get_output('output1')
    assert result == [1, 1]