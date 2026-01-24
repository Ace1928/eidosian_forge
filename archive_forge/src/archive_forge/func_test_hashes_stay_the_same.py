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
@parametrize('to_hash,expected', [('This is a string to hash', '71b3f47df22cb19431d85d92d0b230b2'), (u"C'est lété", '2d8d189e9b2b0b2e384d93c868c0e576'), ((123456, 54321, -98928), 'e205227dd82250871fa25aa0ec690aa3'), ([random.Random(42).random() for _ in range(5)], 'a11ffad81f9682a7d901e6edc3d16c84'), ({'abcde': 123, 'sadfas': [-9999, 2, 3]}, 'aeda150553d4bb5c69f0e69d51b0e2ef')])
def test_hashes_stay_the_same(to_hash, expected):
    assert hash(to_hash) == expected