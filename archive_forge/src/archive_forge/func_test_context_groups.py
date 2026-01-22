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
def test_context_groups():
    surface = ImageSurface(cairocffi.FORMAT_ARGB32, 1, 1)
    context = Context(surface)
    assert isinstance(context.get_target(), ImageSurface)
    assert context.get_target()._pointer == surface._pointer
    assert context.get_group_target()._pointer == surface._pointer
    assert context.get_group_target().get_content() == cairocffi.CONTENT_COLOR_ALPHA
    assert surface.get_data()[:] == pixel(b'\x00\x00\x00\x00')
    with context:
        context.push_group_with_content(cairocffi.CONTENT_ALPHA)
        assert context.get_group_target().get_content() == cairocffi.CONTENT_ALPHA
        context.set_source_rgba(1, 0.2, 0.4, 0.8)
        assert isinstance(context.get_source(), SolidPattern)
        context.paint()
        context.pop_group_to_source()
        assert isinstance(context.get_source(), SurfacePattern)
        assert surface.get_data()[:] == pixel(b'\x00\x00\x00\x00')
        context.paint()
        assert surface.get_data()[:] == pixel(b'\xcc\x00\x00\x00')
    with context:
        context.push_group()
        context.set_source_rgba(1, 0.2, 0.4)
        context.paint()
        group = context.pop_group()
        assert isinstance(context.get_source(), SolidPattern)
        assert isinstance(group, SurfacePattern)
        context.set_source_surface(group.get_surface())
        assert surface.get_data()[:] == pixel(b'\xcc\x00\x00\x00')
        context.paint()
        assert surface.get_data()[:] == pixel(b'\xff\xff3f')