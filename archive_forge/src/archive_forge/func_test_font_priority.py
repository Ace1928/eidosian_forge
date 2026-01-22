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
def test_font_priority():
    with rc_context(rc={'font.sans-serif': ['cmmi10', 'Bitstream Vera Sans']}):
        fontfile = findfont(FontProperties(family=['sans-serif']))
    assert Path(fontfile).name == 'cmmi10.ttf'
    font = get_font(fontfile)
    cmap = font.get_charmap()
    assert len(cmap) == 131
    assert cmap[8729] == 30