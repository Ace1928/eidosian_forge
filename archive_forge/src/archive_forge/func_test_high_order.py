import warnings
from scipy._lib import _pep440
import numpy as np
from numpy.testing import (assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
from numpy import array, spacing, sin, pi, sort, sqrt
from scipy.signal import (argrelextrema, BadCoefficients, bessel, besselap, bilinear,
from scipy.signal._filter_design import (_cplxreal, _cplxpair, _norm_factor,
def test_high_order(self):
    z, p, k = bessel(24, 100, analog=True, output='zpk')
    z2 = []
    p2 = [-90.55312334014323 + 4.844005815403969j, -89.83105162681878 + 14.54056170018573j, -88.37357994162065 + 24.26335240122282j, -86.15278316179575 + 34.03202098404543j, -83.12326467067703 + 43.869859402179j, -79.21695461084202 + 53.80628489700191j, -74.33392285433246 + 63.88084216250878j, -68.32565803501586 + 74.15032695116071j, -60.96221567378025 + 84.70292433074425j, -51.85914574820616 + 95.69048385258847j, -40.27853855197555 + 107.4195196518679j, -24.33481337524861 + 120.7298683731973j]
    k2 = 9.999999999999989e+47
    assert_array_equal(z, z2)
    assert_allclose(sorted(p, key=np.imag), sorted(np.union1d(p2, np.conj(p2)), key=np.imag))
    assert_allclose(k, k2, rtol=1e-14)
    z, p, k = bessel(23, 1000, analog=True, output='zpk')
    z2 = []
    p2 = [-249.7697202208956 + 1202.813187870698j, -412.6986617510172 + 1065.328794475509j, -530.4922463809596 + 943.9760364018479j, -902.7564978975828 + 101.0534334242318j, -890.9283244406079 + 202.3024699647598j, -870.9469394347836 + 303.9581994804637j, -842.380594813137 + 406.2657947488952j, -804.5561642249877 + 509.5305912401127j, -756.466014676626 + 614.1594859516342j, -696.5966033906477 + 720.7341374730186j, -622.5903228776276 + 830.1558302815096j, -906.6732476324988]
    k2 = 9.999999999999983e+68
    assert_array_equal(z, z2)
    assert_allclose(sorted(p, key=np.imag), sorted(np.union1d(p2, np.conj(p2)), key=np.imag))
    assert_allclose(k, k2, rtol=1e-14)
    z, p, k = bessel(31, 1, analog=True, output='zpk', norm='delay')
    p2 = [-20.876706, -20.826543 + 1.735732j, -20.675502 + 3.47332j, -20.421895 + 5.214702j, -20.062802 + 6.961982j, -19.593895 + 8.717546j, -19.009148 + 10.484195j, -18.3004 + 12.265351j, -17.456663 + 14.06535j, -16.463032 + 15.88991j, -15.298849 + 17.746914j, -13.934466 + 19.647827j, -12.324914 + 21.610519j, -10.395893 + 23.665701j, -8.0056 + 25.875019j, -4.792045 + 28.406037j]
    assert_allclose(sorted(p, key=np.imag), sorted(np.union1d(p2, np.conj(p2)), key=np.imag))
    z, p, k = bessel(30, 1, analog=True, output='zpk', norm='delay')
    p2 = [-20.201029 + 0.86775j, -20.097257 + 2.604235j, -19.888485 + 4.343721j, -19.572188 + 6.088363j, -19.14438 + 7.84057j, -18.599342 + 9.603147j, -17.929195 + 11.379494j, -17.123228 + 13.173901j, -16.166808 + 14.992008j, -15.03958 + 16.84158j, -13.712245 + 18.733902j, -12.140295 + 20.686563j, -10.250119 + 22.729808j, -7.90117 + 24.924391j, -4.734679 + 27.435615j]
    assert_allclose(sorted(p, key=np.imag), sorted(np.union1d(p2, np.conj(p2)), key=np.imag))