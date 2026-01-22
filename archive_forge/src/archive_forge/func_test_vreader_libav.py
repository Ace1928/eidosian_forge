import skvideo.io
import sys
import numpy as np
import hashlib
import os
from numpy.testing import assert_equal
def test_vreader_libav():
    if not skvideo._HAS_AVCONV:
        return 0
    try:
        if np.int(skvideo._LIBAV_MAJOR_VERSION) < 12:
            return 0
    except:
        return 0
    _vwrite('libav')