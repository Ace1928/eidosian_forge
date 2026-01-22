import ctypes
from ctypes import wintypes # We can't use ctypes.wintypes, we must import wintypes this way.
from pygetwindow import PyGetWindowException, pointInRect, BaseWindow, Rect, Point, Size
def getWindowsAt(x, y):
    """Returns a list of Window objects whose windows contain the point ``(x, y)``.

    * ``x`` (int, optional): The x position of the window(s).
    * ``y`` (int, optional): The y position of the window(s)."""
    windowsAtXY = []
    for window in getAllWindows():
        if pointInRect(x, y, window.left, window.top, window.width, window.height):
            windowsAtXY.append(window)
    return windowsAtXY