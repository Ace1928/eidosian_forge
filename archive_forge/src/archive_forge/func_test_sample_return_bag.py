from __future__ import annotations
import math
import random as rnd
import pytest
import dask.bag as db
from dask.bag import random
def test_sample_return_bag():
    seq = range(20)
    sut = db.from_sequence(seq, npartitions=3)
    assert isinstance(random.sample(sut, k=2), db.Bag)