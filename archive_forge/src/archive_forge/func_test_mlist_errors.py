import os
import warnings
from pathlib import Path
from unittest import TestCase
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from ..ecat import (
from ..openers import Opener
from ..testing import data_path, suppress_warnings
from ..tmpdirs import InTemporaryDirectory
from . import test_wrapstruct as tws
from .test_fileslice import slicer_samples
def test_mlist_errors(self):
    fid = open(self.example_file, 'rb')
    hdr = self.header_class.from_fileobj(fid)
    hdr['num_frames'] = 6
    mlist = read_mlist(fid, hdr.endianness)
    fid.close()
    mlist = np.array([[16842754.0, 3.0, 12035.0, 1.0], [16842753.0, 12036.0, 24068.0, 1.0], [16842755.0, 24069.0, 36101.0, 1.0], [16842756.0, 36102.0, 48134.0, 1.0], [16842757.0, 48135.0, 60167.0, 1.0], [16842758.0, 60168.0, 72200.0, 1.0]])
    with suppress_warnings():
        series_framenumbers = get_series_framenumbers(mlist)
    assert series_framenumbers[0] == 2
    order = [series_framenumbers[x] for x in sorted(series_framenumbers)]
    assert order == [2, 1, 3, 4, 5, 6]
    mlist[0, 0] = 0
    with suppress_warnings():
        frames_order = get_frame_order(mlist)
    neworder = [frames_order[x][0] for x in sorted(frames_order)]
    assert neworder == [1, 2, 3, 4, 5]
    with suppress_warnings():
        with pytest.raises(OSError):
            get_series_framenumbers(mlist)