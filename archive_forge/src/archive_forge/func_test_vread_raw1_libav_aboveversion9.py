from numpy.testing import assert_equal
import numpy as np
import skvideo
import skvideo.io
import skvideo.utils
import skvideo.datasets
import os
import nose
@nose.tools.nottest
def test_vread_raw1_libav_aboveversion9():
    if not skvideo._HAS_AVCONV:
        return 0
    if np.int(skvideo._LIBAV_MAJOR_VERSION) < 9:
        return 0
    _rawhelper1('libav')