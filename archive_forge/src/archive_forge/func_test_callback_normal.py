from time import sleep
import pytest
import nipype.interfaces.utility as niu
import nipype.pipeline.engine as pe
@pytest.mark.parametrize('plugin', ['Linear', 'MultiProc', 'LegacyMultiProc'])
def test_callback_normal(tmpdir, plugin):
    tmpdir.chdir()
    so = Status()
    wf = pe.Workflow(name='test', base_dir=tmpdir.strpath)
    f_node = pe.Node(niu.Function(function=func, input_names=[], output_names=[]), name='f_node')
    wf.add_nodes([f_node])
    wf.config['execution'] = {'crashdump_dir': wf.base_dir, 'poll_sleep_duration': 2}
    wf.run(plugin=plugin, plugin_args={'status_callback': so.callback})
    assert so.statuses == [('f_node', 'start'), ('f_node', 'end')]