import pytest
from ....testing import example_data
from ...niftyreg import get_custom_path
from ...niftyreg.tests.test_regutils import no_nifty_tool
from .. import UnaryMaths, BinaryMaths, BinaryMathsInteger, TupleMaths, Merge
@pytest.mark.skipif(no_nifty_tool(cmd='seg_maths'), reason='niftyseg is not installed')
def test_binary_maths():
    binarym = BinaryMaths()
    cmd = get_custom_path('seg_maths', env_dir='NIFTYSEGDIR')
    assert binarym.cmd == cmd
    with pytest.raises(ValueError):
        binarym.run()
    in_file = example_data('im1.nii')
    binarym.inputs.in_file = in_file
    binarym.inputs.operand_value = 2.0
    binarym.inputs.operation = 'sub'
    binarym.inputs.output_datatype = 'float'
    cmd_tmp = '{cmd} {in_file} -sub 2.00000000 -odt float {out_file}'
    expected_cmd = cmd_tmp.format(cmd=cmd, in_file=in_file, out_file='im1_sub.nii')
    assert binarym.cmdline == expected_cmd