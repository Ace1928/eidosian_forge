import pytest
import numpy as np
import ase.io.ulm as ulm
def test_ulm(ulmfile):
    with ulm.open(ulmfile) as r:
        assert r.y == 9
        assert r.s == 'abc'
        assert (A.read(r.a).x == np.ones((2, 3))).all()
        assert (r.a.x == np.ones((2, 3))).all()
        assert r[1].s == 'abc2'
        assert r[2].s == 'abc3'
        assert (r[2].z == np.ones(7)).all()