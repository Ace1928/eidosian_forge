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
def test_context_font():
    surface = ImageSurface(cairocffi.FORMAT_ARGB32, 10, 10)
    context = Context._from_pointer(Context(surface)._pointer, incref=True)
    assert context.get_font_matrix().as_tuple() == (10, 0, 0, 10, 0, 0)
    context.set_font_matrix(Matrix(2, 0, 0, 3, 12, 4))
    assert context.get_font_matrix().as_tuple() == (2, 0, 0, 3, 12, 4)
    context.set_font_size(14)
    assert context.get_font_matrix().as_tuple() == (14, 0, 0, 14, 0, 0)
    context.set_font_size(10)
    context.select_font_face(b'@cairo:serif', cairocffi.FONT_SLANT_ITALIC)
    font_face = context.get_font_face()
    assert isinstance(font_face, ToyFontFace)
    assert font_face.get_family() == '@cairo:serif'
    assert font_face.get_slant() == cairocffi.FONT_SLANT_ITALIC
    assert font_face.get_weight() == cairocffi.FONT_WEIGHT_NORMAL
    try:
        del cairocffi.fonts.FONT_TYPE_TO_CLASS[cairocffi.FONT_TYPE_TOY]
        re_font_face = context.get_font_face()
        assert re_font_face._pointer == font_face._pointer
        assert isinstance(re_font_face, FontFace)
        assert not isinstance(re_font_face, ToyFontFace)
    finally:
        cairocffi.fonts.FONT_TYPE_TO_CLASS[cairocffi.FONT_TYPE_TOY] = ToyFontFace
    ascent, descent, height, max_x_advance, max_y_advance = context.font_extents()
    assert max_x_advance > 0
    assert max_y_advance == 0
    _, _, _, _, x_advance, y_advance = context.text_extents('i' * 10)
    assert x_advance > 0
    assert y_advance == 0
    context.set_font_face(ToyFontFace('@cairo:monospace', weight=cairocffi.FONT_WEIGHT_BOLD))
    _, _, _, _, x_advance_mono, y_advance = context.text_extents('i' * 10)
    assert x_advance_mono > x_advance
    assert y_advance == 0
    assert list(context.copy_path()) == []
    context.text_path('a')
    assert list(context.copy_path())
    assert surface.get_data()[:] == b'\x00' * 400
    context.move_to(1, 9)
    context.show_text('a')
    assert surface.get_data()[:] != b'\x00' * 400
    assert context.get_font_options().get_hint_metrics() == cairocffi.HINT_METRICS_DEFAULT
    context.set_font_options(FontOptions(hint_metrics=cairocffi.HINT_METRICS_ON))
    assert context.get_font_options().get_hint_metrics() == cairocffi.HINT_METRICS_ON
    assert surface.get_font_options().get_hint_metrics() == cairocffi.HINT_METRICS_ON
    context.set_font_matrix(Matrix(2, 0, 0, 3, 12, 4))
    assert context.get_scaled_font().get_font_matrix().as_tuple() == (2, 0, 0, 3, 12, 4)
    context.set_scaled_font(ScaledFont(ToyFontFace(), font_matrix=Matrix(0, 1, 4, 0, 12, 4)))
    assert context.get_font_matrix().as_tuple() == (0, 1, 4, 0, 12, 4)
    context.set_font_face(None)