from os.path import basename, isfile, join
from unittest.mock import MagicMock, Mock, patch
import numpy
from numpy import array as npa
from numpy.testing import assert_almost_equal, assert_array_equal
import nibabel
from nibabel.cmdline import parrec2nii
from nibabel.tests.test_parrec import EG_PAR, VARY_PAR
from nibabel.tmpdirs import InTemporaryDirectory
@patch('nibabel.cmdline.parrec2nii.verbose')
def test_parrec2nii_save_load_qform_code(*args):
    parrec2nii.verbose.switch = False
    opts = Mock()
    opts.outdir = None
    opts.scaling = 'off'
    opts.minmax = [1, 1]
    opts.store_header = False
    opts.bvs = False
    opts.vol_info = False
    opts.dwell_time = False
    opts.compressed = False
    with InTemporaryDirectory() as pth:
        opts.outdir = pth
        for fname in [EG_PAR, VARY_PAR]:
            parrec2nii.proc_file(fname, opts)
            outfname = join(pth, basename(fname)).replace('.PAR', '.nii')
            assert isfile(outfname)
            img = nibabel.load(outfname)
            assert_almost_equal(img.affine, PAR_AFFINE, 4)
            assert img.header['qform_code'] == 1
            assert_array_equal(img.header['sform_code'], 1)