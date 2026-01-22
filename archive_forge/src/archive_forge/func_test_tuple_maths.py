import pytest
from ....testing import example_data
from ...niftyreg import get_custom_path
from ...niftyreg.tests.test_regutils import no_nifty_tool
from .. import UnaryMaths, BinaryMaths, BinaryMathsInteger, TupleMaths, Merge
@pytest.mark.skipif(no_nifty_tool(cmd='seg_maths'), reason='niftyseg is not installed')
def test_tuple_maths():
    tuplem = TupleMaths()
    cmd = get_custom_path('seg_maths', env_dir='NIFTYSEGDIR')
    assert tuplem.cmd == cmd
    with pytest.raises(ValueError):
        tuplem.run()
    in_file = example_data('im1.nii')
    op_file = example_data('im2.nii')
    tuplem.inputs.in_file = in_file
    tuplem.inputs.operation = 'lncc'
    tuplem.inputs.operand_file1 = op_file
    tuplem.inputs.operand_value2 = 2.0
    tuplem.inputs.output_datatype = 'float'
    cmd_tmp = '{cmd} {in_file} -lncc {op} 2.00000000 -odt float {out_file}'
    expected_cmd = cmd_tmp.format(cmd=cmd, in_file=in_file, op=op_file, out_file='im1_lncc.nii')
    assert tuplem.cmdline == expected_cmd