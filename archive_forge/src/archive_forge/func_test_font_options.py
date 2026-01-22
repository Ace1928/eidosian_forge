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
def test_font_options():
    options = FontOptions()
    assert options.get_antialias() == cairocffi.ANTIALIAS_DEFAULT
    options.set_antialias(cairocffi.ANTIALIAS_FAST)
    assert options.get_antialias() == cairocffi.ANTIALIAS_FAST
    assert options.get_subpixel_order() == cairocffi.SUBPIXEL_ORDER_DEFAULT
    options.set_subpixel_order(cairocffi.SUBPIXEL_ORDER_BGR)
    assert options.get_subpixel_order() == cairocffi.SUBPIXEL_ORDER_BGR
    assert options.get_hint_style() == cairocffi.HINT_STYLE_DEFAULT
    options.set_hint_style(cairocffi.HINT_STYLE_SLIGHT)
    assert options.get_hint_style() == cairocffi.HINT_STYLE_SLIGHT
    assert options.get_hint_metrics() == cairocffi.HINT_METRICS_DEFAULT
    options.set_hint_metrics(cairocffi.HINT_METRICS_OFF)
    assert options.get_hint_metrics() == cairocffi.HINT_METRICS_OFF
    options_1 = FontOptions(hint_metrics=cairocffi.HINT_METRICS_ON)
    assert options_1.get_hint_metrics() == cairocffi.HINT_METRICS_ON
    assert options_1.get_antialias() == cairocffi.HINT_METRICS_DEFAULT
    options_2 = options_1.copy()
    assert options_2 == options_1
    assert len(set([options_1, options_2])) == 1
    options_2.set_antialias(cairocffi.ANTIALIAS_BEST)
    assert options_2 != options_1
    assert len(set([options_1, options_2])) == 2
    options_1.merge(options_2)
    assert options_2 == options_1