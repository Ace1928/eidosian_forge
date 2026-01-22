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
def test_addfont_as_path():
    """Smoke test that addfont() accepts pathlib.Path."""
    font_test_file = 'mpltest.ttf'
    path = Path(__file__).parent / font_test_file
    try:
        fontManager.addfont(path)
        added, = [font for font in fontManager.ttflist if font.fname.endswith(font_test_file)]
        fontManager.ttflist.remove(added)
    finally:
        to_remove = [font for font in fontManager.ttflist if font.fname.endswith(font_test_file)]
        for font in to_remove:
            fontManager.ttflist.remove(font)