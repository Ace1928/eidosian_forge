import os
import pytest
import nipype.interfaces.fsl.model as fsl
from nipype.interfaces.fsl import no_fsl
from pathlib import Path
from ....pipeline import engine as pe
@pytest.mark.skipif(no_fsl(), reason='fsl is not installed')
def test_MultipleRegressDesign(tmpdir):
    designer = pe.Node(fsl.MultipleRegressDesign(), name='designer', base_dir=str(tmpdir))
    designer.inputs.regressors = dict(voice_stenght=[1, 1, 1], age=[0.2, 0.4, 0.5], BMI=[1, -1, 2])
    con1 = ['voice_and_age', 'T', ['age', 'voice_stenght'], [0.5, 0.5]]
    con2 = ['just_BMI', 'T', ['BMI'], [1]]
    designer.inputs.contrasts = [con1, con2, ['con3', 'F', [con1, con2]], ['con4', 'F', [con2]]]
    res = designer.run()
    outputs = res.outputs.get_traitsfree()
    for ftype in ['mat', 'con', 'fts', 'grp']:
        assert Path(outputs['design_' + ftype]).exists()
    expected_content = {}
    expected_content['design_mat'] = '/NumWaves       3\n/NumPoints      3\n/PPheights      3.000000e+00 5.000000e-01 1.000000e+00\n\n/Matrix\n1.000000e+00 2.000000e-01 1.000000e+00\n-1.000000e+00 4.000000e-01 1.000000e+00\n2.000000e+00 5.000000e-01 1.000000e+00\n'
    expected_content['design_con'] = '/ContrastName1   voice_and_age\n/ContrastName2   just_BMI\n/NumWaves       3\n/NumContrasts   2\n/PPheights          1.000000e+00 1.000000e+00\n/RequiredEffect     100.000 100.000\n\n/Matrix\n0.000000e+00 5.000000e-01 5.000000e-01\n1.000000e+00 0.000000e+00 0.000000e+00\n'
    expected_content['design_fts'] = '/NumWaves       2\n/NumContrasts   2\n\n/Matrix\n1 1\n0 1\n'
    expected_content['design_grp'] = '/NumWaves       1\n/NumPoints      3\n\n/Matrix\n1\n1\n1\n'
    for ftype in ['mat', 'con', 'fts', 'grp']:
        outfile = 'design_' + ftype
        assert Path(outputs[outfile]).read_text() == expected_content[outfile]