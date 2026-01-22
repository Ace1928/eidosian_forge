from __future__ import absolute_import
import pytest
from ..symbolic import SymbolicSys
from ..util import requires, import_
from .test_symbolic import decay_dydt_factory
@requires('sym', 'scipy')
def test_banded_jacobian():
    k = [4, 3]
    odesys = SymbolicSys.from_callback(decay_dydt_factory(k), len(k) + 1)
    bj = odesys.be.banded_jacobian(odesys.exprs, odesys.dep, 1, 0)
    assert bj.tolist() == [[-k[0], -k[1], 0], [k[0], k[1], 0]]