from numpy.testing import (assert_, assert_equal, assert_almost_equal,
from pytest import raises as assert_raises
import pytest
from numpy import mgrid, pi, sin, ogrid, poly1d, linspace
import numpy as np
from scipy.interpolate import (interp1d, interp2d, lagrange, PPoly, BPoly,
from scipy.special import poch, gamma
from scipy.interpolate import _ppoly
from scipy._lib._gcutils import assert_deallocated, IS_PYPY
from scipy.integrate import nquad
from scipy.special import binom
def test_validation(self):
    for kind in ('nearest', 'nearest-up', 'zero', 'linear', 'slinear', 'quadratic', 'cubic', 'previous', 'next'):
        interp1d(self.x10, self.y10, kind=kind)
        interp1d(self.x10, self.y10, kind=kind, fill_value='extrapolate')
    interp1d(self.x10, self.y10, kind='linear', fill_value=(-1, 1))
    interp1d(self.x10, self.y10, kind='linear', fill_value=np.array([-1]))
    interp1d(self.x10, self.y10, kind='linear', fill_value=(-1,))
    interp1d(self.x10, self.y10, kind='linear', fill_value=-1)
    interp1d(self.x10, self.y10, kind='linear', fill_value=(-1, -1))
    interp1d(self.x10, self.y10, kind=0)
    interp1d(self.x10, self.y10, kind=1)
    interp1d(self.x10, self.y10, kind=2)
    interp1d(self.x10, self.y10, kind=3)
    interp1d(self.x10, self.y210, kind='linear', axis=-1, fill_value=(-1, -1))
    interp1d(self.x2, self.y210, kind='linear', axis=0, fill_value=np.ones(10))
    interp1d(self.x2, self.y210, kind='linear', axis=0, fill_value=(np.ones(10), np.ones(10)))
    interp1d(self.x2, self.y210, kind='linear', axis=0, fill_value=(np.ones(10), -1))
    assert_raises(ValueError, interp1d, self.x25, self.y10)
    assert_raises(ValueError, interp1d, self.x10, np.array(0))
    assert_raises(ValueError, interp1d, self.x10, self.y2)
    assert_raises(ValueError, interp1d, self.x2, self.y10)
    assert_raises(ValueError, interp1d, self.x10, self.y102)
    interp1d(self.x10, self.y210)
    interp1d(self.x10, self.y102, axis=0)
    assert_raises(ValueError, interp1d, self.x1, self.y10)
    assert_raises(ValueError, interp1d, self.x10, self.y1)
    assert_raises(ValueError, interp1d, self.x10, self.y10, kind='linear', fill_value=(-1, -1, -1))
    assert_raises(ValueError, interp1d, self.x10, self.y10, kind='linear', fill_value=[-1, -1, -1])
    assert_raises(ValueError, interp1d, self.x10, self.y10, kind='linear', fill_value=np.array((-1, -1, -1)))
    assert_raises(ValueError, interp1d, self.x10, self.y10, kind='linear', fill_value=[[-1]])
    assert_raises(ValueError, interp1d, self.x10, self.y10, kind='linear', fill_value=[-1, -1])
    assert_raises(ValueError, interp1d, self.x10, self.y10, kind='linear', fill_value=np.array([]))
    assert_raises(ValueError, interp1d, self.x10, self.y10, kind='linear', fill_value=())
    assert_raises(ValueError, interp1d, self.x2, self.y210, kind='linear', axis=0, fill_value=[-1, -1])
    assert_raises(ValueError, interp1d, self.x2, self.y210, kind='linear', axis=0, fill_value=(0.0, [-1, -1]))