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
def test_hash_numpy_dict_of_arrays(three_np_arrays):
    arr1, arr2, arr3 = three_np_arrays
    d1 = {1: arr1, 2: arr2}
    d2 = {1: arr2, 2: arr1}
    d3 = {1: arr2, 2: arr3}
    assert hash(d1) == hash(d2)
    assert hash(d1) != hash(d3)