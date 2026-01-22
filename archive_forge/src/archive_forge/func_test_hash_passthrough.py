import unittest
import os
import sys
import subprocess
from collections import defaultdict
from textwrap import dedent
import numpy as np
from numba import jit, config, typed, typeof
from numba.core import types, utils
import unittest
from numba.tests.support import (TestCase, skip_unless_py10_or_later,
from numba.cpython.unicode import compile_time_get_string_data
from numba.cpython import hashing
def test_hash_passthrough(self):
    kind1_string = 'abcdefghijklmnopqrstuvwxyz'

    @jit(nopython=True)
    def fn(x):
        return x._hash
    hash_value = compile_time_get_string_data(kind1_string)[-1]
    self.assertTrue(hash_value != -1)
    self.assertEqual(fn(kind1_string), hash_value)