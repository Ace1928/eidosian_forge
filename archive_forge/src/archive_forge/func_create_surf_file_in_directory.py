import os
import pytest
import numpy as np
import nibabel as nb
from nipype.utils.filemanip import ensure_list
from nipype.interfaces.fsl import Info
from nipype.interfaces.fsl.base import FSLCommand
@pytest.fixture()
def create_surf_file_in_directory(request, tmpdir):
    cwd = tmpdir.chdir()
    surf = 'lh.a.nii'
    nifti_image_files(tmpdir.strpath, filelist=surf, shape=(1, 100, 1))

    def change_directory():
        cwd.chdir()
    request.addfinalizer(change_directory)
    return (surf, tmpdir.strpath)