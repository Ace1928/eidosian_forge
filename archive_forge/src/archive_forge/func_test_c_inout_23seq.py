import os
import sys
import copy
import platform
import pytest
import numpy as np
from numpy.testing import assert_, assert_equal
from numpy.core.multiarray import typeinfo as _typeinfo
from . import util
def test_c_inout_23seq(self):
    obj = np.array(self.num23seq, dtype=self.type.dtype)
    shape = (len(self.num23seq), len(self.num23seq[0]))
    a = self.array(shape, intent.in_.c.inout, obj)
    assert a.has_shared_memory()