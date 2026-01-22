import time
import hashlib
import sys
import gc
import io
import collections
import itertools
import pickle
import random
from concurrent.futures import ProcessPoolExecutor
from decimal import Decimal
from joblib.hashing import hash
from joblib.func_inspect import filter_args
from joblib.memory import Memory
from joblib.testing import raises, skipif, fixture, parametrize
from joblib.test.common import np, with_numpy
@with_numpy
def test_numpy_dtype_pickling():
    dt1 = np.dtype('f4')
    dt2 = np.dtype('f4')
    assert dt1 is dt2
    assert hash(dt1) == hash(dt2)
    dt1_roundtripped = pickle.loads(pickle.dumps(dt1))
    assert dt1 is not dt1_roundtripped
    assert hash(dt1) == hash(dt1_roundtripped)
    assert hash([dt1, dt1]) == hash([dt1_roundtripped, dt1_roundtripped])
    assert hash([dt1, dt1]) == hash([dt1, dt1_roundtripped])
    complex_dt1 = np.dtype([('name', np.str_, 16), ('grades', np.float64, (2,))])
    complex_dt2 = np.dtype([('name', np.str_, 16), ('grades', np.float64, (2,))])
    assert hash(complex_dt1) == hash(complex_dt2)
    complex_dt1_roundtripped = pickle.loads(pickle.dumps(complex_dt1))
    assert complex_dt1_roundtripped is not complex_dt1
    assert hash(complex_dt1) == hash(complex_dt1_roundtripped)
    assert hash([complex_dt1, complex_dt1]) == hash([complex_dt1_roundtripped, complex_dt1_roundtripped])
    assert hash([complex_dt1, complex_dt1]) == hash([complex_dt1_roundtripped, complex_dt1])