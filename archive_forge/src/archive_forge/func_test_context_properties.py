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
def test_context_properties():
    surface = ImageSurface(cairocffi.FORMAT_ARGB32, 1, 1)
    context = Context(surface)
    assert context.get_antialias() == cairocffi.ANTIALIAS_DEFAULT
    context.set_antialias(cairocffi.ANTIALIAS_BEST)
    assert context.get_antialias() == cairocffi.ANTIALIAS_BEST
    assert context.get_dash() == ([], 0)
    context.set_dash([4, 1, 3, 2], 1.5)
    assert context.get_dash() == ([4, 1, 3, 2], 1.5)
    assert context.get_dash_count() == 4
    assert context.get_fill_rule() == cairocffi.FILL_RULE_WINDING
    context.set_fill_rule(cairocffi.FILL_RULE_EVEN_ODD)
    assert context.get_fill_rule() == cairocffi.FILL_RULE_EVEN_ODD
    assert context.get_line_cap() == cairocffi.LINE_CAP_BUTT
    context.set_line_cap(cairocffi.LINE_CAP_SQUARE)
    assert context.get_line_cap() == cairocffi.LINE_CAP_SQUARE
    assert context.get_line_join() == cairocffi.LINE_JOIN_MITER
    context.set_line_join(cairocffi.LINE_JOIN_ROUND)
    assert context.get_line_join() == cairocffi.LINE_JOIN_ROUND
    assert context.get_line_width() == 2
    context.set_line_width(13)
    assert context.get_line_width() == 13
    assert context.get_miter_limit() == 10
    context.set_miter_limit(4)
    assert context.get_miter_limit() == 4
    assert context.get_operator() == cairocffi.OPERATOR_OVER
    context.set_operator(cairocffi.OPERATOR_XOR)
    assert context.get_operator() == cairocffi.OPERATOR_XOR
    assert context.get_tolerance() == 0.1
    context.set_tolerance(0.25)
    assert context.get_tolerance() == 0.25