import os
import os.path as op
import pytest
from nipype.testing.fixtures import (
from nipype.pipeline import engine as pe
from nipype.interfaces import freesurfer as fs
from nipype.interfaces.base import TraitError
from nipype.interfaces.io import FreeSurferSource
@pytest.mark.skipif(fs.no_freesurfer(), reason='freesurfer is not installed')
def test_surfsmooth(create_surf_file_in_directory):
    smooth = fs.SurfaceSmooth()
    assert smooth.cmd == 'mri_surf2surf'
    with pytest.raises(ValueError):
        smooth.run()
    surf, cwd = create_surf_file_in_directory
    smooth.inputs.in_file = surf
    smooth.inputs.subject_id = 'fsaverage'
    fwhm = 5
    smooth.inputs.fwhm = fwhm
    smooth.inputs.hemi = 'lh'
    assert smooth.cmdline == 'mri_surf2surf --cortex --fwhm 5.0000 --hemi lh --sval %s --tval %s/lh.a_smooth%d.nii --s fsaverage' % (surf, cwd, fwhm)
    shmooth = fs.SurfaceSmooth(subject_id='fsaverage', fwhm=6, in_file=surf, hemi='lh', out_file='lh.a_smooth.nii')
    assert smooth != shmooth