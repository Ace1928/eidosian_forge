import os
import sys
import copy
import platform
import pytest
import numpy as np
from numpy.testing import assert_, assert_equal
from numpy.core.multiarray import typeinfo as _typeinfo
from . import util
def test_c_copy_in_from_23casttype(self):
    for t in self.type.cast_types():
        obj = np.array(self.num23seq, dtype=t.dtype)
        a = self.array([len(self.num23seq), len(self.num23seq[0])], intent.in_.c.copy, obj)
        assert not a.has_shared_memory()