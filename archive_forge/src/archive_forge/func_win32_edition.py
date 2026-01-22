import collections
import os
import re
import sys
import functools
import itertools
def win32_edition():
    try:
        try:
            import winreg
        except ImportError:
            import _winreg as winreg
    except ImportError:
        pass
    else:
        try:
            cvkey = 'SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion'
            with winreg.OpenKeyEx(winreg.HKEY_LOCAL_MACHINE, cvkey) as key:
                return winreg.QueryValueEx(key, 'EditionId')[0]
        except OSError:
            pass
    return None