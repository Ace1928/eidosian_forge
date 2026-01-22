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
def test_issue_868(self):
    """
        Summary: multiplying a scalar by a non-scalar would cause a crash in
        type inference because TimeDeltaMixOp always assumed at least one of
        its operands was an NPTimeDelta in its generic() method.
        """
    with self.assertRaises(TypingError) as raises:
        njit((types.Array(types.int32, 1, 'C'),))(issue_868)
    expected = (_header_lead + ' Function(<built-in function mul>) found for signature:\n \n >>> mul(UniTuple({} x 1), {})').format(str(types.intp), types.IntegerLiteral(2))
    self.assertIn(expected, str(raises.exception))
    self.assertIn('During: typing of', str(raises.exception))