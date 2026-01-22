from io import BytesIO, StringIO
import gc
import multiprocessing
import os
from pathlib import Path
from PIL import Image
import shutil
import subprocess
import sys
import warnings
import numpy as np
import pytest
from matplotlib.font_manager import (
from matplotlib import cbook, ft2font, pyplot as plt, rc_context, figure as mfigure
@pytest.mark.skipif(sys.platform != 'linux' or not has_fclist, reason='only Linux with fontconfig installed')
def test_user_fonts_linux(tmpdir, monkeypatch):
    font_test_file = 'mpltest.ttf'
    fonts = findSystemFonts()
    if any((font_test_file in font for font in fonts)):
        pytest.skip(f'{font_test_file} already exists in system fonts')
    user_fonts_dir = tmpdir.join('fonts')
    user_fonts_dir.ensure(dir=True)
    shutil.copyfile(Path(__file__).parent / font_test_file, user_fonts_dir.join(font_test_file))
    with monkeypatch.context() as m:
        m.setenv('XDG_DATA_HOME', str(tmpdir))
        _get_fontconfig_fonts.cache_clear()
        fonts = findSystemFonts()
        assert any((font_test_file in font for font in fonts))
    _get_fontconfig_fonts.cache_clear()