import unittest
from ctypes import *
import _ctypes_test
def test_structures(self):
    WNDPROC = WINFUNCTYPE(c_long, c_int, c_int, c_int, c_int)

    def wndproc(hwnd, msg, wParam, lParam):
        return hwnd + msg + wParam + lParam
    HINSTANCE = c_int
    HICON = c_int
    HCURSOR = c_int
    LPCTSTR = c_char_p

    class WNDCLASS(Structure):
        _fields_ = [('style', c_uint), ('lpfnWndProc', WNDPROC), ('cbClsExtra', c_int), ('cbWndExtra', c_int), ('hInstance', HINSTANCE), ('hIcon', HICON), ('hCursor', HCURSOR), ('lpszMenuName', LPCTSTR), ('lpszClassName', LPCTSTR)]
    wndclass = WNDCLASS()
    wndclass.lpfnWndProc = WNDPROC(wndproc)
    WNDPROC_2 = WINFUNCTYPE(c_long, c_int, c_int, c_int, c_int)
    self.assertIs(WNDPROC, WNDPROC_2)
    self.assertEqual(wndclass.lpfnWndProc(1, 2, 3, 4), 10)
    f = wndclass.lpfnWndProc
    del wndclass
    del wndproc
    self.assertEqual(f(10, 11, 12, 13), 46)