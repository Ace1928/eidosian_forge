import pytest
from ..base import FSSurfaceCommand
from ... import freesurfer as fs
from ...io import FreeSurferSource
@pytest.mark.skipif(fs.no_freesurfer(), reason='freesurfer is not installed')
def test_associated_file(tmpdir):
    fssrc = FreeSurferSource(subjects_dir=fs.Info.subjectsdir(), subject_id='fsaverage', hemi='lh')
    fssrc.base_dir = tmpdir.strpath
    fssrc.resource_monitor = False
    fsavginfo = fssrc.run().outputs.get()
    for white, pial in [('lh.white', 'lh.pial'), ('./lh.white', './lh.pial'), (fsavginfo['white'], fsavginfo['pial'])]:
        for name in ('pial', 'lh.pial', pial):
            assert FSSurfaceCommand._associated_file(white, name) == pial
        for name in ('./pial', './lh.pial', fsavginfo['pial']):
            assert FSSurfaceCommand._associated_file(white, name) == name