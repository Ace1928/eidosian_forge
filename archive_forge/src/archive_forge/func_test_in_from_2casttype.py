import os
import sys
import copy
import platform
import pytest
import numpy as np
from numpy.testing import assert_, assert_equal
from numpy.core.multiarray import typeinfo as _typeinfo
from . import util
def test_in_from_2casttype(self):
    for t in self.type.cast_types():
        obj = np.array(self.num2seq, dtype=t.dtype)
        a = self.array([len(self.num2seq)], intent.in_, obj)
        if t.elsize == self.type.elsize:
            assert a.has_shared_memory(), repr((self.type.dtype, t.dtype))
        else:
            assert not a.has_shared_memory()