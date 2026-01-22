from nipype.interfaces.ants import (
import os
import pytest
def test_WarpImageMultiTransform_invaffine_2(change_dir, create_wimt):
    wimt = create_wimt
    wimt.inputs.invert_affine = [2]
    assert wimt.cmdline == 'WarpImageMultiTransform 3 diffusion_weighted.nii diffusion_weighted_wimt.nii -R functional.nii func2anat_coreg_Affine.txt func2anat_InverseWarp.nii.gz dwi2anat_Warp.nii.gz -i dwi2anat_coreg_Affine.txt'