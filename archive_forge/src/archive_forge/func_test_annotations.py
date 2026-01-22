import ctypes
import itertools
import pickle
import random
import typing as pt
import unittest
from collections import OrderedDict
import numpy as np
from numba import (boolean, deferred_type, float32, float64, int16, int32,
from numba.core import errors, types
from numba.core.dispatcher import Dispatcher
from numba.core.errors import LoweringError, TypingError
from numba.core.runtime.nrt import MemInfo
from numba.experimental import jitclass
from numba.experimental.jitclass import _box
from numba.experimental.jitclass.base import JitClassType
from numba.tests.support import MemoryLeakMixin, TestCase, skip_if_typeguard
from numba.tests.support import skip_unless_scipy
@skip_if_typeguard
def test_annotations(self):
    """
        Methods with annotations should compile fine (issue #1911).
        """
    from .annotation_usecases import AnnotatedClass
    spec = {'x': int32}
    cls = jitclass(AnnotatedClass, spec)
    obj = cls(5)
    self.assertEqual(obj.x, 5)
    self.assertEqual(obj.add(2), 7)