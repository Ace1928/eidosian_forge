from __future__ import annotations
import math
import random as rnd
import pytest
import dask.bag as db
from dask.bag import random
def test_choices_size_exactly_k():
    seq = range(20)
    sut = db.from_sequence(seq, npartitions=3)
    li = list(random.choices(sut, k=2).compute())
    assert len(li) == 2
    assert all((i in seq for i in li))