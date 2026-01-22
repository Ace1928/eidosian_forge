import ctypes
from ctypes import wintypes # We can't use ctypes.wintypes, we must import wintypes this way.
from pygetwindow import PyGetWindowException, pointInRect, BaseWindow, Rect, Point, Size
def getAllWindows():
    """Returns a list of Window objects for all visible windows.
    """
    windowObjs = []

    def foreach_window(hWnd, lParam):
        if ctypes.windll.user32.IsWindowVisible(hWnd) != 0:
            windowObjs.append(Win32Window(hWnd))
        return True
    enumWindows(enumWindowsProc(foreach_window), 0)
    return windowObjs