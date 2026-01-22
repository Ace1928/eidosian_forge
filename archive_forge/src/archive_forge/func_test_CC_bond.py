import numpy as np
import pytest
from ase import Atoms
from ase.calculators.bond_polarizability import BondPolarizability
from ase.calculators.bond_polarizability import LippincottStuttman, Linearized
def test_CC_bond():
    """Test polarizabilties of a single CC bond"""
    C2 = Atoms('C2', positions=[[0, 0, 0], [0, 0, 1.69]])

    def check_symmetry(alpha):
        alpha_diag = np.diagonal(alpha)
        assert alpha == pytest.approx(np.diag(alpha_diag))
        assert alpha_diag[0] == alpha_diag[1]
    bp = BondPolarizability()
    check_symmetry(bp(C2))
    bp = BondPolarizability(Linearized())
    check_symmetry(bp(C2))