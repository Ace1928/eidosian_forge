import io
import sys
import textwrap
from test.support import warnings_helper, captured_stdout, captured_stderr
import traceback
import unittest
from unittest.util import strclass
def bad_cleanup2():
    print('do cleanup2')
    raise ValueError('bad cleanup2')