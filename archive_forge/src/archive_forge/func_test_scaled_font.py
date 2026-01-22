import array
import base64
import contextlib
import gc
import io
import math
import os
import shutil
import sys
import tempfile
import cairocffi
import pikepdf
import pytest
from . import (
def test_scaled_font():
    font = ScaledFont(ToyFontFace())
    font_extents = font.extents()
    ascent, descent, height, max_x_advance, max_y_advance = font_extents
    assert max_x_advance > 0
    assert max_y_advance == 0
    _, _, _, _, x_advance, y_advance = font.text_extents('i' * 10)
    assert x_advance > 0
    assert y_advance == 0
    font = ScaledFont(ToyFontFace('@cairo:serif'))
    _, _, _, _, x_advance, y_advance = font.text_extents('i' * 10)
    font = ScaledFont(ToyFontFace('@cairo:monospace'))
    _, _, _, _, x_advance_mono, y_advance = font.text_extents('i' * 10)
    assert x_advance_mono > x_advance
    assert y_advance == 0
    assert isinstance(font.get_font_face(), FontFace)
    font = ScaledFont(ToyFontFace('@cairo:monospace'), Matrix(xx=20, yy=20), Matrix(xx=3, yy=0.5), FontOptions(antialias=cairocffi.ANTIALIAS_BEST))
    assert font.get_font_options().get_antialias() == cairocffi.ANTIALIAS_BEST
    assert font.get_font_matrix().as_tuple() == (20, 0, 0, 20, 0, 0)
    assert font.get_ctm().as_tuple() == (3, 0, 0, 0.5, 0, 0)
    assert font.get_scale_matrix().as_tuple() == (60, 0, 0, 10, 0, 0)
    _, _, _, _, x_advance_mono_2, y_advance_2 = font.text_extents('i' * 10)
    assert y_advance == y_advance_2
    assert x_advance_mono_2 > x_advance_mono