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
@pytest.mark.xfail(cairo_version() < 11000, reason='Cairo version too low')
def test_mime_data():
    surface = PDFSurface(None, 1, 1)
    assert surface.get_mime_data('image/jpeg') is None
    gc.collect()
    assert len(cairocffi.surfaces.KeepAlive.instances) == 0
    surface.set_mime_data('image/jpeg', b'lol')
    assert len(cairocffi.surfaces.KeepAlive.instances) == 1
    assert surface.get_mime_data('image/jpeg')[:] == b'lol'
    surface.set_mime_data('image/jpeg', None)
    assert len(cairocffi.surfaces.KeepAlive.instances) == 0
    if cairo_version() >= 11200:
        assert surface.get_mime_data('image/jpeg') is None
    surface.finish()
    assert_raise_finished(surface.set_mime_data, 'image/jpeg', None)