import pytest
from ....testing import example_data
from ...niftyreg import get_custom_path
from ...niftyreg.tests.test_regutils import no_nifty_tool
from .. import UnaryMaths, BinaryMaths, BinaryMathsInteger, TupleMaths, Merge
@pytest.mark.skipif(no_nifty_tool(cmd='seg_maths'), reason='niftyseg is not installed')
def test_unary_maths():
    unarym = UnaryMaths()
    cmd = get_custom_path('seg_maths', env_dir='NIFTYSEGDIR')
    assert unarym.cmd == cmd
    with pytest.raises(ValueError):
        unarym.run()
    in_file = example_data('im1.nii')
    unarym.inputs.in_file = in_file
    unarym.inputs.operation = 'otsu'
    unarym.inputs.output_datatype = 'float'
    expected_cmd = '{cmd} {in_file} -otsu -odt float {out_file}'.format(cmd=cmd, in_file=in_file, out_file='im1_otsu.nii')
    assert unarym.cmdline == expected_cmd