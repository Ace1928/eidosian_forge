import random
from mpmath import *
from mpmath.libmp import *
def test_mpmathify():
    assert mpmathify('1/2') == 0.5
    assert mpmathify('(1.0+1.0j)') == mpc(1, 1)
    assert mpmathify('(1.2e-10 - 3.4e5j)') == mpc('1.2e-10', '-3.4e5')
    assert mpmathify('1j') == mpc(1j)