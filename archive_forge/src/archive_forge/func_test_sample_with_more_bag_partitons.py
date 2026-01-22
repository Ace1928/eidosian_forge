from __future__ import annotations
import math
import random as rnd
import pytest
import dask.bag as db
from dask.bag import random
def test_sample_with_more_bag_partitons():
    seq = range(100)
    sut = db.from_sequence(seq, npartitions=10)
    li = list(random.sample(sut, k=10, split_every=8).compute())
    assert sut.map_partitions(len).compute() == (10, 10, 10, 10, 10, 10, 10, 10, 10, 10)
    assert len(li) == 10
    assert all((i in seq for i in li))
    assert len(set(li)) == len(li)