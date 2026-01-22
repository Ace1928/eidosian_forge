from __future__ import absolute_import, division, print_function
import collections
import sys
import time
import datetime
import os
import platform
import re
import functools
from contextlib import contextmanager
@_genericPyAutoGUIChecks
def mouseDown(x=None, y=None, button=PRIMARY, duration=0.0, tween=linear, logScreenshot=None, _pause=True):
    """Performs pressing a mouse button down (but not up).

    The x and y parameters detail where the mouse event happens. If None, the
    current mouse position is used. If a float value, it is rounded down. If
    outside the boundaries of the screen, the event happens at edge of the
    screen.

    Args:
      x (int, float, None, tuple, optional): The x position on the screen where the
        mouse down happens. None by default. If tuple, this is used for x and y.
        If x is a str, it's considered a filename of an image to find on
        the screen with locateOnScreen() and click the center of.
      y (int, float, None, optional): The y position on the screen where the
        mouse down happens. None by default.
      button (str, int, optional): The mouse button pressed down. TODO

    Returns:
      None

    Raises:
      PyAutoGUIException: If button is not one of 'left', 'middle', 'right', 1, 2, or 3
    """
    button = _normalizeButton(button)
    x, y = _normalizeXYArgs(x, y)
    _mouseMoveDrag('move', x, y, 0, 0, duration=0, tween=None)
    _logScreenshot(logScreenshot, 'mouseDown', '%s,%s' % (x, y), folder='.')
    platformModule._mouseDown(x, y, button)