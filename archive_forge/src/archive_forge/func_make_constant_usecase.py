import numpy as np
import unittest
from numba import jit, vectorize, int8, int16, int32
from numba.tests.support import TestCase
from numba.tests.enum_usecases import (Color, Shape, Shake,
def make_constant_usecase(const):

    def constant_usecase(a):
        return a is const
    return constant_usecase