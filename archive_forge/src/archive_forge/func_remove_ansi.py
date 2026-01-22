import sys
import re
import unittest
from curtsies.fmtfuncs import bold, green, magenta, cyan, red, plain
from unittest import mock
from bpython.curtsiesfrontend import interpreter
def remove_ansi(s):
    return re.sub('(\\x9B|\\x1B\\[)[0-?]*[ -\\/]*[@-~]'.encode('ascii'), b'', s)