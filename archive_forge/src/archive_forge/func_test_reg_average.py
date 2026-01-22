import os
import pytest
from ....utils.filemanip import which
from ....testing import example_data
from .. import (
@pytest.mark.skipif(no_nifty_tool(cmd='reg_average'), reason='niftyreg is not installed. reg_average not found.')
def test_reg_average():
    """tests for reg_average interface"""
    nr_average = RegAverage()
    assert nr_average.cmd == get_custom_path('reg_average')
    one_file = example_data('im1.nii')
    two_file = example_data('im2.nii')
    three_file = example_data('im3.nii')
    nr_average.inputs.avg_files = [one_file, two_file, three_file]
    nr_average.inputs.omp_core_val = 1
    generated_cmd = nr_average.cmdline
    reg_average_cmd = os.path.join(os.getcwd(), 'reg_average_cmd')
    with open(reg_average_cmd, 'rb') as f_obj:
        argv = f_obj.read()
    os.remove(reg_average_cmd)
    expected_argv = '%s %s -avg %s %s %s -omp 1' % (get_custom_path('reg_average'), os.path.join(os.getcwd(), 'avg_out.nii.gz'), one_file, two_file, three_file)
    assert argv.decode('utf-8') == expected_argv
    expected_cmd = '%s --cmd_file %s' % (get_custom_path('reg_average'), reg_average_cmd)
    assert generated_cmd == expected_cmd
    nr_average_2 = RegAverage()
    one_file = example_data('TransformParameters.0.txt')
    two_file = example_data('ants_Affine.txt')
    three_file = example_data('elastix.txt')
    nr_average_2.inputs.avg_files = [one_file, two_file, three_file]
    nr_average_2.inputs.omp_core_val = 1
    generated_cmd = nr_average_2.cmdline
    reg_average_cmd = os.path.join(os.getcwd(), 'reg_average_cmd')
    with open(reg_average_cmd, 'rb') as f_obj:
        argv = f_obj.read()
    os.remove(reg_average_cmd)
    expected_argv = '%s %s -avg %s %s %s -omp 1' % (get_custom_path('reg_average'), os.path.join(os.getcwd(), 'avg_out.txt'), one_file, two_file, three_file)
    assert argv.decode('utf-8') == expected_argv
    nr_average_3 = RegAverage()
    one_file = example_data('TransformParameters.0.txt')
    two_file = example_data('ants_Affine.txt')
    three_file = example_data('elastix.txt')
    nr_average_3.inputs.avg_lts_files = [one_file, two_file, three_file]
    nr_average_3.inputs.omp_core_val = 1
    generated_cmd = nr_average_3.cmdline
    reg_average_cmd = os.path.join(os.getcwd(), 'reg_average_cmd')
    with open(reg_average_cmd, 'rb') as f_obj:
        argv = f_obj.read()
    os.remove(reg_average_cmd)
    expected_argv = '%s %s -avg_lts %s %s %s -omp 1' % (get_custom_path('reg_average'), os.path.join(os.getcwd(), 'avg_out.txt'), one_file, two_file, three_file)
    assert argv.decode('utf-8') == expected_argv
    nr_average_4 = RegAverage()
    ref_file = example_data('anatomical.nii')
    one_file = example_data('im1.nii')
    two_file = example_data('im2.nii')
    three_file = example_data('im3.nii')
    trans1_file = example_data('roi01.nii')
    trans2_file = example_data('roi02.nii')
    trans3_file = example_data('roi03.nii')
    nr_average_4.inputs.warp_files = [trans1_file, one_file, trans2_file, two_file, trans3_file, three_file]
    nr_average_4.inputs.avg_ref_file = ref_file
    nr_average_4.inputs.omp_core_val = 1
    generated_cmd = nr_average_4.cmdline
    reg_average_cmd = os.path.join(os.getcwd(), 'reg_average_cmd')
    with open(reg_average_cmd, 'rb') as f_obj:
        argv = f_obj.read()
    os.remove(reg_average_cmd)
    expected_argv = '%s %s -avg_tran %s -omp 1 %s %s %s %s %s %s' % (get_custom_path('reg_average'), os.path.join(os.getcwd(), 'avg_out.nii.gz'), ref_file, trans1_file, one_file, trans2_file, two_file, trans3_file, three_file)
    assert argv.decode('utf-8') == expected_argv
    nr_average_5 = RegAverage()
    ref_file = example_data('anatomical.nii')
    one_file = example_data('im1.nii')
    two_file = example_data('im2.nii')
    three_file = example_data('im3.nii')
    aff1_file = example_data('TransformParameters.0.txt')
    aff2_file = example_data('ants_Affine.txt')
    aff3_file = example_data('elastix.txt')
    trans1_file = example_data('roi01.nii')
    trans2_file = example_data('roi02.nii')
    trans3_file = example_data('roi03.nii')
    nr_average_5.inputs.warp_files = [aff1_file, trans1_file, one_file, aff2_file, trans2_file, two_file, aff3_file, trans3_file, three_file]
    nr_average_5.inputs.demean3_ref_file = ref_file
    nr_average_5.inputs.omp_core_val = 1
    generated_cmd = nr_average_5.cmdline
    reg_average_cmd = os.path.join(os.getcwd(), 'reg_average_cmd')
    with open(reg_average_cmd, 'rb') as f_obj:
        argv = f_obj.read()
    os.remove(reg_average_cmd)
    expected_argv = '%s %s -demean3 %s -omp 1 %s %s %s %s %s %s %s %s %s' % (get_custom_path('reg_average'), os.path.join(os.getcwd(), 'avg_out.nii.gz'), ref_file, aff1_file, trans1_file, one_file, aff2_file, trans2_file, two_file, aff3_file, trans3_file, three_file)
    assert argv.decode('utf-8') == expected_argv