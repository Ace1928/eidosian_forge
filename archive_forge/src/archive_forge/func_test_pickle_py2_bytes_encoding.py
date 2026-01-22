import copy
import sys
import gc
import tempfile
import pytest
from os import path
from io import BytesIO
from itertools import chain
import numpy as np
from numpy.testing import (
from numpy.testing._private.utils import _no_tracing, requires_memory
from numpy.compat import asbytes, asunicode, pickle
def test_pickle_py2_bytes_encoding(self):
    test_data = [(np.str_('漬'), b"cnumpy.core.multiarray\nscalar\np0\n(cnumpy\ndtype\np1\n(S'U1'\np2\nI0\nI1\ntp3\nRp4\n(I3\nS'<'\np5\nNNNI4\nI4\nI0\ntp6\nbS',o\\x00\\x00'\np7\ntp8\nRp9\n."), (np.array([9e+123], dtype=np.float64), b"cnumpy.core.multiarray\n_reconstruct\np0\n(cnumpy\nndarray\np1\n(I0\ntp2\nS'b'\np3\ntp4\nRp5\n(I1\n(I1\ntp6\ncnumpy\ndtype\np7\n(S'f8'\np8\nI0\nI1\ntp9\nRp10\n(I3\nS'<'\np11\nNNNI-1\nI-1\nI0\ntp12\nbI00\nS'O\\x81\\xb7Z\\xaa:\\xabY'\np13\ntp14\nb."), (np.array([(9e+123,)], dtype=[('name', float)]), b"cnumpy.core.multiarray\n_reconstruct\np0\n(cnumpy\nndarray\np1\n(I0\ntp2\nS'b'\np3\ntp4\nRp5\n(I1\n(I1\ntp6\ncnumpy\ndtype\np7\n(S'V8'\np8\nI0\nI1\ntp9\nRp10\n(I3\nS'|'\np11\nN(S'name'\np12\ntp13\n(dp14\ng12\n(g7\n(S'f8'\np15\nI0\nI1\ntp16\nRp17\n(I3\nS'<'\np18\nNNNI-1\nI-1\nI0\ntp19\nbI0\ntp20\nsI8\nI1\nI0\ntp21\nbI00\nS'O\\x81\\xb7Z\\xaa:\\xabY'\np22\ntp23\nb.")]
    for original, data in test_data:
        result = pickle.loads(data, encoding='bytes')
        assert_equal(result, original)
        if isinstance(result, np.ndarray) and result.dtype.names is not None:
            for name in result.dtype.names:
                assert_(isinstance(name, str))