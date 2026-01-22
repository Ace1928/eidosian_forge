import ctypes
from ctypes import wintypes # We can't use ctypes.wintypes, we must import wintypes this way.
from pygetwindow import PyGetWindowException, pointInRect, BaseWindow, Rect, Point, Size
def getAllTitles():
    """Returns a list of strings of window titles for all visible windows.
    """
    return [window.title for window in getAllWindows()]