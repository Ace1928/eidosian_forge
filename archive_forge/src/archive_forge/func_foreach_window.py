import ctypes
from ctypes import wintypes # We can't use ctypes.wintypes, we must import wintypes this way.
from pygetwindow import PyGetWindowException, pointInRect, BaseWindow, Rect, Point, Size
def foreach_window(hWnd, lParam):
    if ctypes.windll.user32.IsWindowVisible(hWnd) != 0:
        windowObjs.append(Win32Window(hWnd))
    return True