import os
import time
import pytest
import xcffib.xproto  # noqa isort:skip
from xcffib.xproto import ConfigWindow, CW, EventMask, GC  # noqa isort:skip
from . import Context, XCBSurface, cairo_version  # noqa isort:skip
@pytest.fixture
def xcb_conn():
    """
    Fixture that will setup and take down a xcffib.Connection object running on
    a display spawned by xvfb
    """
    display = os.environ.get('DISPLAY')
    if display is None:
        pytest.skip('DISPLAY environment variable not set')
    conn = xcffib.connect(display)
    yield conn
    conn.disconnect()