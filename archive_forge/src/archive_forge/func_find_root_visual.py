import os
import time
import pytest
import xcffib.xproto  # noqa isort:skip
from xcffib.xproto import ConfigWindow, CW, EventMask, GC  # noqa isort:skip
from . import Context, XCBSurface, cairo_version  # noqa isort:skip
def find_root_visual(conn):
    """Find the xcffib.xproto.VISUALTYPE corresponding to the root visual"""
    default_screen = conn.setup.roots[conn.pref_screen]
    for i in default_screen.allowed_depths:
        for v in i.visuals:
            if v.visual_id == default_screen.root_visual:
                return v