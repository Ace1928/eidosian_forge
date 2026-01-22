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
@pytest.mark.xfail(cairo_version() < 11400, reason='Cairo version too low')
def test_device_scale():
    surface = PDFSurface(None, 1, 1)
    assert surface.get_device_scale() == (1, 1)
    surface.set_device_scale(2, 3)
    assert surface.get_device_scale() == (2, 3)