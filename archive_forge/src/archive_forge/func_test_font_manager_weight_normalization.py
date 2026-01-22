from io import BytesIO
import pytest
import logging
from matplotlib import _afm
from matplotlib import font_manager as fm
def test_font_manager_weight_normalization():
    font = _afm.AFM(BytesIO(AFM_TEST_DATA.replace(b'Weight Bold\n', b'Weight Custom\n')))
    assert fm.afmFontProperty('', font).weight == 'normal'