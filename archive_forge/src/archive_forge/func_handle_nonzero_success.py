import builtins
import ctypes.wintypes
from paramiko.util import u
def handle_nonzero_success(result):
    if result == 0:
        raise WindowsError()