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
def create_objects_to_hash():
    rng = np.random.RandomState(42)
    to_hash_list = [rng.randint(-1000, high=1000, size=50).astype('<i8'), tuple((rng.randn(3).astype('<f4') for _ in range(5))), [rng.randn(3).astype('<f4') for _ in range(5)], {-3333: rng.randn(3, 5).astype('<f4'), 0: [rng.randint(10, size=20).astype('<i8'), rng.randn(10).astype('<f4')]}, np.arange(100, dtype='<i8').reshape((10, 10)), np.asfortranarray(np.arange(100, dtype='<i8').reshape((10, 10))), np.arange(100, dtype='<i8').reshape((10, 10))[:, :2]]
    return to_hash_list