import math
import re
import textwrap
import operator
import numpy as np
import unittest
from numba import jit, njit
from numba.core import types
from numba.core.errors import TypingError
from numba.core.types.functions import _header_lead
from numba.tests.support import TestCase
def test_template_rejection_error_message_cascade(self):
    from numba import njit

    @njit
    def foo():
        z = 1
        for a, b in enumerate(z):
            pass
        return z
    with self.assertRaises(TypingError) as raises:
        foo()
    errmsg = str(raises.exception)
    expected = 'No match.'
    self.assertIn(expected, errmsg)
    ctx_lines = [x for x in errmsg.splitlines() if 'During:' in x]
    search = ['.*During: resolving callee type: Function.*enumerate', '.*During: typing of call .*test_typingerror.py']
    for i, x in enumerate(search):
        self.assertTrue(re.search(x, ctx_lines[i]))