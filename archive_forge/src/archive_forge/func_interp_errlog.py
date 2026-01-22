import sys
import re
import unittest
from curtsies.fmtfuncs import bold, green, magenta, cyan, red, plain
from unittest import mock
from bpython.curtsiesfrontend import interpreter
def interp_errlog(self):
    i = interpreter.Interp()
    a = []
    i.write = a.append
    return (i, a)