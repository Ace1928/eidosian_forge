import itertools
import functools
import sys
import operator
from collections import namedtuple
import numpy as np
import unittest
import warnings
from numba import jit, typeof, njit, typed
from numba.core import errors, types, config
from numba.tests.support import (TestCase, tag, ignore_internal_warnings,
from numba.core.extending import overload_method, box
def test_isinstance_exceptions(self):
    fns = [(invalid_isinstance_usecase, 'Cannot infer numba type of python type'), (invalid_isinstance_usecase_phi_nopropagate, 'isinstance() cannot determine the type of variable "z" due to a branch.'), (invalid_isinstance_optional_usecase, 'isinstance() cannot determine the type of variable "z" due to a branch.'), (invalid_isinstance_unsupported_type_usecase(), 'isinstance() does not support variables of type "ntpl(')]
    for fn, msg in fns:
        fn = njit(fn)
        with self.assertRaises(errors.TypingError) as raises:
            fn(100)
        self.assertIn(msg, str(raises.exception))