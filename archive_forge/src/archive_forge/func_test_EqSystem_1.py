import collections
import pytest
from ..util.testing import requires
from ..chemistry import Substance, Reaction, Equilibrium, Species
@requires('numpy')
def test_EqSystem_1():
    es = _get_es1()
    assert es.stoichs().tolist() == [[-1, 1]]
    assert es.eq_constants() == [3]