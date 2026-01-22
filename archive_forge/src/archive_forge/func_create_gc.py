import os
import time
import pytest
import xcffib.xproto  # noqa isort:skip
from xcffib.xproto import ConfigWindow, CW, EventMask, GC  # noqa isort:skip
from . import Context, XCBSurface, cairo_version  # noqa isort:skip
def create_gc(conn):
    """Creates a simple graphics context"""
    gc = conn.generate_id()
    default_screen = conn.setup.roots[conn.pref_screen]
    conn.core.CreateGC(gc, default_screen.root, GC.Foreground | GC.Background, [default_screen.black_pixel, default_screen.white_pixel])
    return gc