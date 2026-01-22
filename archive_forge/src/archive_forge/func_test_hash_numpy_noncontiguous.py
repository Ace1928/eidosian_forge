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
def test_hash_numpy_noncontiguous():
    a = np.asarray(np.arange(6000).reshape((1000, 2, 3)), order='F')[:, :1, :]
    b = np.ascontiguousarray(a)
    assert hash(a) != hash(b)
    c = np.asfortranarray(a)
    assert hash(a) != hash(c)