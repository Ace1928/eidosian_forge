import math
import unittest
from textwrap import dedent
import pytest
import affine
from affine import Affine, EPSILON
def seq_almost_equal(t1, t2, error=1e-05):
    assert len(t1) == len(t2), f'{t1!r} != {t2!r}'
    for m1, m2 in zip(t1, t2):
        assert abs(m1 - m2) <= error, f'{t1!r} != {t2!r}'