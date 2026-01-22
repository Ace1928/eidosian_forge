import pytest
from pytest import raises as assert_raises
import numpy as np
from scipy.cluster.hierarchy import DisjointSet
import string
def test_element_not_present():
    elements = get_elements(n=10)
    dis = DisjointSet(elements)
    with assert_raises(KeyError):
        dis['dummy']
    with assert_raises(KeyError):
        dis.merge(elements[0], 'dummy')
    with assert_raises(KeyError):
        dis.connected(elements[0], 'dummy')