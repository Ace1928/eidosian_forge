import os
import nipype.interfaces.fsl.dti as fsl
from nipype.interfaces.fsl import Info, no_fsl
from nipype.interfaces.base import Undefined
import pytest
from nipype.testing.fixtures import create_files_in_directory
@pytest.mark.skipif(no_fsl(), reason='fsl is not installed')
def test_distancemap(create_files_in_directory):
    mapper = fsl.DistanceMap()
    files, newdir = create_files_in_directory
    assert mapper.cmd == 'distancemap'
    with pytest.raises(ValueError):
        mapper.run()
    mapper.inputs.in_file = 'a.nii'
    assert mapper.cmdline == 'distancemap --out=%s --in=a.nii' % os.path.join(newdir, 'a_dstmap.nii')
    mapper.inputs.local_max_file = True
    assert mapper.cmdline == 'distancemap --out=%s --in=a.nii --localmax=%s' % (os.path.join(newdir, 'a_dstmap.nii'), os.path.join(newdir, 'a_lclmax.nii'))
    mapper.inputs.local_max_file = 'max.nii'
    assert mapper.cmdline == 'distancemap --out=%s --in=a.nii --localmax=max.nii' % os.path.join(newdir, 'a_dstmap.nii')