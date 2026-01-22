import os
import pytest
from ....utils.filemanip import which
from ....testing import example_data
from .. import (
@pytest.mark.skipif(no_nifty_tool(cmd='reg_resample'), reason='niftyreg is not installed. reg_resample not found.')
def test_reg_resample_res():
    """tests for reg_resample interface"""
    nr_resample = RegResample()
    assert nr_resample.cmd == get_custom_path('reg_resample')
    with pytest.raises(ValueError):
        nr_resample.run()
    ref_file = example_data('im1.nii')
    flo_file = example_data('im2.nii')
    trans_file = example_data('warpfield.nii')
    nr_resample.inputs.ref_file = ref_file
    nr_resample.inputs.flo_file = flo_file
    nr_resample.inputs.trans_file = trans_file
    nr_resample.inputs.inter_val = 'LIN'
    nr_resample.inputs.omp_core_val = 4
    cmd_tmp = '{cmd} -flo {flo} -inter 1 -omp 4 -ref {ref} -trans {trans} -res {res}'
    expected_cmd = cmd_tmp.format(cmd=get_custom_path('reg_resample'), flo=flo_file, ref=ref_file, trans=trans_file, res='im2_res.nii.gz')
    assert nr_resample.cmdline == expected_cmd
    nr_resample_2 = RegResample(type='blank', inter_val='LIN', omp_core_val=4)
    ref_file = example_data('im1.nii')
    flo_file = example_data('im2.nii')
    trans_file = example_data('warpfield.nii')
    nr_resample_2.inputs.ref_file = ref_file
    nr_resample_2.inputs.flo_file = flo_file
    nr_resample_2.inputs.trans_file = trans_file
    cmd_tmp = '{cmd} -flo {flo} -inter 1 -omp 4 -ref {ref} -trans {trans} -blank {blank}'
    expected_cmd = cmd_tmp.format(cmd=get_custom_path('reg_resample'), flo=flo_file, ref=ref_file, trans=trans_file, blank='im2_blank.nii.gz')
    assert nr_resample_2.cmdline == expected_cmd