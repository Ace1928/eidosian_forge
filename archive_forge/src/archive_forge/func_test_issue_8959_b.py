import functools
import unittest
from test import support
from ctypes import *
from ctypes.test import need_symbol
from _ctypes import CTYPES_MAX_ARGCOUNT
import _ctypes_test
@need_symbol('WINFUNCTYPE')
def test_issue_8959_b(self):
    from ctypes.wintypes import BOOL, HWND, LPARAM
    global windowCount
    windowCount = 0

    @WINFUNCTYPE(BOOL, HWND, LPARAM)
    def EnumWindowsCallbackFunc(hwnd, lParam):
        global windowCount
        windowCount += 1
        return True
    windll.user32.EnumWindows(EnumWindowsCallbackFunc, 0)