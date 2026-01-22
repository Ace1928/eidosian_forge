from os.path import join as pjoin
import numpy as np
import pytest
from .. import minc2
from ..minc2 import Minc2File, Minc2Image
from ..optpkg import optional_package
from ..testing import data_path
from . import test_minc1 as tm2
def test_bad_diminfo():
    fname = pjoin(data_path, 'minc2_baddim.mnc')
    with pytest.warns(UserWarning) as w:
        Minc2Image.from_filename(fname)