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
def check_hash_values(self, values):
    cfunc = self.cfunc
    for val in list(values):
        nb_hash = cfunc(val)
        self.assertIsInstance(nb_hash, int)
        try:
            self.assertEqual(nb_hash, hash(val))
        except AssertionError as e:
            print('val, nb_hash, hash(val)')
            print(val, nb_hash, hash(val))
            print('abs(val), hashing._PyHASH_MODULUS - 1')
            print(abs(val), hashing._PyHASH_MODULUS - 1)
            raise e