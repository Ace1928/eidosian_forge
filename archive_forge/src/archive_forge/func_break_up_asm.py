import ctypes
import threading
from ctypes import CFUNCTYPE, c_int, c_int32
from ctypes.util import find_library
import gc
import locale
import os
import platform
import re
import subprocess
import sys
import unittest
from contextlib import contextmanager
from tempfile import mkstemp
from llvmlite import ir
from llvmlite import binding as llvm
from llvmlite.binding import ffi
from llvmlite.tests import TestCase
def break_up_asm(self, asm):
    asm_list = []
    for line in asm.splitlines():
        s_line = line.strip()
        if not (s_line.startswith('.') or s_line.startswith('fpadd') or s_line == ''):
            asm_list.append(s_line)
    return asm_list