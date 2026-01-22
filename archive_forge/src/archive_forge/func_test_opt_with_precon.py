import numpy as np
import pytest
from ase.build import molecule
from ase.utils.ff import Morse, Angle, Dihedral, VdW
from ase.calculators.ff import ForceField
from ase.optimize.precon.neighbors import get_neighbours
from ase.optimize.precon.lbfgs import PreconLBFGS
from ase.optimize.precon import FF
@pytest.mark.slow
def test_opt_with_precon(atoms, forcefield_params):
    kw = dict(forcefield_params)
    kw.pop('vdws')
    precon = FF(**kw)
    opt = PreconLBFGS(atoms, use_armijo=True, precon=precon)
    opt.run(fmax=0.1)
    e = atoms.get_potential_energy()
    assert abs(e - ref_energy) < 0.01