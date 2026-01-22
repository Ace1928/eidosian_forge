import itertools
import numpy as np
from numba import jit
from numba.core import utils
from numba.tests.support import TestCase, forbid_codegen
from .enum_usecases import *
import unittest
def test_float_values(self):
    for tp in self.float_types:
        for prec in ['exact', 'single', 'double']:
            self.eq(tp(1.5), tp(1.5), prec=prec)
            self.eq(tp(0.0), tp(0.0), prec=prec)
            self.eq(tp(-0.0), tp(-0.0), prec=prec)
            self.ne(tp(0.0), tp(-0.0), prec=prec)
            self.eq(tp(0.0), tp(-0.0), prec=prec, ignore_sign_on_zero=True)
            self.eq(tp(INF), tp(INF), prec=prec)
            self.ne(tp(INF), tp(1e+38), prec=prec)
            self.eq(tp(-INF), tp(-INF), prec=prec)
            self.ne(tp(INF), tp(-INF), prec=prec)
            self.eq(tp(NAN), tp(NAN), prec=prec)
            self.ne(tp(NAN), tp(0), prec=prec)
            self.ne(tp(NAN), tp(INF), prec=prec)
            self.ne(tp(NAN), tp(-INF), prec=prec)