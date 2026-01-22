import numpy as np
import unittest
from numba import jit, vectorize, int8, int16, int32
from numba.tests.support import TestCase
from numba.tests.enum_usecases import (Color, Shape, Shake,
def identity_usecase(a, b, c):
    return (a is Shake.mint, b is Shape.circle, c is RequestError.internal_error)