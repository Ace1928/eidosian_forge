import collections
import pytest
from ..util.testing import requires
from ..chemistry import Substance, Reaction, Equilibrium, Species
@pytest.mark.xfail
@requires('numpy')
@pytest.mark.parametrize('NumSys', [(NumSysLin,), (NumSysLog,), (NumSysLog, NumSysLin)])
def test_precipitate(NumSys):
    eqsys, species, cases = _get_NaCl(Species, phase_idx=1)
    for init, final in cases:
        x, sol, sane = eqsys.root(dict(zip(species, init)), NumSys=NumSys, rref_preserv=True, tol=1e-12)
        assert sol['success'] and sane
        assert x is not None
        assert np.allclose(x, np.asarray(final))