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
def test_unknown_function(self):
    try:
        njit(())(foo)
    except TypingError as e:
        self.assertIn("Untyped global name 'what'", str(e))
    else:
        self.fail('Should raise error')