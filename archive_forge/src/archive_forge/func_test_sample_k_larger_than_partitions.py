from __future__ import annotations
import math
import random as rnd
import pytest
import dask.bag as db
from dask.bag import random
def test_sample_k_larger_than_partitions():
    bag = db.from_sequence(range(10), partition_size=3)
    bag2 = random.sample(bag, k=8, split_every=2)
    seq = bag2.compute()
    assert len(seq) == 8