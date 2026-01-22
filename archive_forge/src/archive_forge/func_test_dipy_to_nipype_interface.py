import pytest
from packaging.version import Version
from collections import namedtuple
from ...base import traits, File, TraitedSpec, BaseInterfaceInputSpec
from ..base import (
@pytest.mark.skipif(no_dipy(), reason='DIPY is not installed')
def test_dipy_to_nipype_interface():
    from dipy.workflows.workflow import Workflow

    class DummyWorkflow(Workflow):

        @classmethod
        def get_short_name(cls):
            return 'dwf1'

        def run(self, in_files, param1=1, out_dir='', out_ref='out1.txt'):
            """Workflow used to test basic workflows.

            Parameters
            ----------
            in_files : string
                fake input string param
            param1 : int, optional
                fake positional param (default 1)
            out_dir : string, optional
                fake output directory (default '')
            out_ref : string, optional
                fake out file (default out1.txt)

            References
            -----------
            dummy references

            """
            return param1
    new_specs = dipy_to_nipype_interface('MyModelSpec', DummyWorkflow)
    assert new_specs.__base__ == DipyBaseInterface
    assert isinstance(new_specs(), DipyBaseInterface)
    assert new_specs.__name__ == 'MyModelSpec'
    assert hasattr(new_specs, 'input_spec')
    assert new_specs().input_spec.__base__ == BaseInterfaceInputSpec
    assert hasattr(new_specs, 'output_spec')
    assert new_specs().output_spec.__base__ == TraitedSpec
    assert hasattr(new_specs, '_run_interface')
    assert hasattr(new_specs, '_list_outputs')
    params_in = new_specs().inputs.get()
    params_out = new_specs()._outputs().get()
    assert len(params_in) == 4
    assert 'in_files' in params_in.keys()
    assert 'param1' in params_in.keys()
    assert 'out_dir' in params_out.keys()
    assert 'out_ref' in params_out.keys()
    with pytest.raises(ValueError):
        new_specs().run()