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
def test_context_stroke():
    for preserve in [True, False]:
        surface = ImageSurface(cairocffi.FORMAT_A8, 4, 4)
        assert surface.get_data()[:] == b'\x00' * 16
        context = Context(surface)
        context.set_source_rgba(0, 0, 0, 1)
        context.set_line_width(1)
        context.rectangle(0.5, 0.5, 2, 2)
        path = list(context.copy_path())
        assert path
        context.stroke_preserve() if preserve else context.stroke()
        assert list(context.copy_path()) == (path if preserve else [])
        assert surface.get_data()[:] == b'\xff\xff\xff\x00\xff\x00\xff\x00\xff\xff\xff\x00\x00\x00\x00\x00'