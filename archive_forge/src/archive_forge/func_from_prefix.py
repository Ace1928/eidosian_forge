import sys
import os
import os.path
import ctypes
from ctypes import c_char_p, c_int, c_size_t, c_void_p, pointer, CFUNCTYPE, POINTER
import ctypes.util
import platform
import textwrap
def from_prefix(prefix):
    find_message('finding from prefix ', prefix)
    assert os.path.exists(prefix), prefix + '  does not exist'
    bin_path = os.path.join(prefix, 'bin')
    enchant_dll_path = os.path.join(bin_path, 'libenchant-2.dll')
    assert os.path.exists(enchant_dll_path), enchant_dll_path + ' does not exist'
    new_path = bin_path + os.pathsep + os.environ['PATH']
    find_message('Prepending ', bin_path, ' to %PATH%')
    os.environ['PATH'] = new_path
    return enchant_dll_path