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
def test_hash_methods():
    a = io.StringIO(unicode('a'))
    assert hash(a.flush) == hash(a.flush)
    a1 = collections.deque(range(10))
    a2 = collections.deque(range(9))
    assert hash(a1.extend) != hash(a2.extend)