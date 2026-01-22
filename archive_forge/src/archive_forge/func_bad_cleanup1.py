import io
import sys
import textwrap
from test.support import warnings_helper, captured_stdout, captured_stderr
import traceback
import unittest
from unittest.util import strclass
def bad_cleanup1():
    print('do cleanup1')
    raise TypeError('bad cleanup1')