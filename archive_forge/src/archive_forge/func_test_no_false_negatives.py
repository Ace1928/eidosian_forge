from __future__ import division
import pytest
import pickle
from preshed.bloom import BloomFilter
def test_no_false_negatives():
    bf = BloomFilter(size=100, hash_funcs=2)
    for ii in range(0, 1000, 20):
        bf.add(ii)
    for ii in range(0, 1000, 20):
        assert ii in bf