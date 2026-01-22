import ctypes
from ctypes import wintypes # We can't use ctypes.wintypes, we must import wintypes this way.
from pygetwindow import PyGetWindowException, pointInRect, BaseWindow, Rect, Point, Size
def getActiveWindow():
    """Returns a Window object of the currently active (focused) Window."""
    hWnd = ctypes.windll.user32.GetForegroundWindow()
    if hWnd == 0:
        return None
    else:
        return Win32Window(hWnd)