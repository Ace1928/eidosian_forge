import contextlib
import ctypes
from ctypes import (
import os
import platform
from shutil import which as _executable_exists
import subprocess
import time
import warnings
from pandas.errors import (
from pandas.util._exceptions import find_stack_level
def determine_clipboard():
    """
    Determine the OS/platform and set the copy() and paste() functions
    accordingly.
    """
    global Foundation, AppKit, qtpy, PyQt4, PyQt5
    if 'cygwin' in platform.system().lower():
        if os.path.exists('/dev/clipboard'):
            warnings.warn("Pyperclip's support for Cygwin is not perfect, see https://github.com/asweigart/pyperclip/issues/55", stacklevel=find_stack_level())
            return init_dev_clipboard_clipboard()
    elif os.name == 'nt' or platform.system() == 'Windows':
        return init_windows_clipboard()
    if platform.system() == 'Linux':
        if _executable_exists('wslconfig.exe'):
            return init_wsl_clipboard()
    if os.name == 'mac' or platform.system() == 'Darwin':
        try:
            import AppKit
            import Foundation
        except ImportError:
            return init_osx_pbcopy_clipboard()
        else:
            return init_osx_pyobjc_clipboard()
    if HAS_DISPLAY:
        if os.environ.get('WAYLAND_DISPLAY') and _executable_exists('wl-copy'):
            return init_wl_clipboard()
        if _executable_exists('xsel'):
            return init_xsel_clipboard()
        if _executable_exists('xclip'):
            return init_xclip_clipboard()
        if _executable_exists('klipper') and _executable_exists('qdbus'):
            return init_klipper_clipboard()
        try:
            import qtpy
        except ImportError:
            try:
                import PyQt5
            except ImportError:
                try:
                    import PyQt4
                except ImportError:
                    pass
                else:
                    return init_qt_clipboard()
            else:
                return init_qt_clipboard()
        else:
            return init_qt_clipboard()
    return init_no_clipboard()