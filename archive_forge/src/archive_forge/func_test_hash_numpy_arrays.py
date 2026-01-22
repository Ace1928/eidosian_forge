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
def test_hash_numpy_arrays(three_np_arrays):
    arr1, arr2, arr3 = three_np_arrays
    for obj1, obj2 in itertools.product(three_np_arrays, repeat=2):
        are_hashes_equal = hash(obj1) == hash(obj2)
        are_arrays_equal = np.all(obj1 == obj2)
        assert are_hashes_equal == are_arrays_equal
    assert hash(arr1) != hash(arr1.T)