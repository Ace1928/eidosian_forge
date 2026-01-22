import numpy as np
import pytest
from ase.build import bulk
from ase.constraints import FixAtoms, UnitCellFilter
from ase.calculators.emt import EMT
from ase.optimize.precon import make_precon, Precon
from ase.neighborlist import neighbor_list
from ase.utils.ff import Bond
def check_assembly(precon, system):
    atoms, bonds = system
    kwargs = {}
    if precon == 'FF' or precon == 'Exp_FF':
        kwargs['bonds'] = bonds
    precon = make_precon(precon, atoms, **kwargs)
    assert isinstance(precon, Precon)
    N = 3 * len(atoms)
    P = precon.asarray()
    assert P.shape == (N, N)
    assert np.abs(P - P.T).max() < 1e-06
    assert np.all(np.linalg.eigvalsh(P)) > 0