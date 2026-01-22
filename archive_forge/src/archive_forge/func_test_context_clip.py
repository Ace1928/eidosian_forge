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
def test_context_clip():
    surface = ImageSurface(cairocffi.FORMAT_A8, 4, 4)
    assert surface.get_data()[:] == b'\x00' * 16
    context = Context(surface)
    context.rectangle(1, 1, 2, 2)
    assert context.clip_extents() == (0, 0, 4, 4)
    path = list(context.copy_path())
    assert path
    context.clip_preserve()
    assert list(context.copy_path()) == path
    assert context.clip_extents() == (1, 1, 3, 3)
    context.clip()
    assert list(context.copy_path()) == []
    assert context.clip_extents() == (1, 1, 3, 3)
    context.reset_clip()
    assert context.clip_extents() == (0, 0, 4, 4)
    context.rectangle(1, 1, 2, 2)
    context.rectangle(1, 2, 1, 2)
    context.clip()
    assert context.copy_clip_rectangle_list() == [(1, 1, 2, 2), (1, 3, 1, 1)]
    assert context.clip_extents() == (1, 1, 3, 4)