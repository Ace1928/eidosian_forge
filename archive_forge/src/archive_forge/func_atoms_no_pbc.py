from functools import partial
import pytest
from ase.calculators.emt import EMT
from ase.optimize import (MDMin, FIRE, LBFGS, LBFGSLineSearch, BFGSLineSearch,
from ase.optimize.sciopt import SciPyFminCG, SciPyFminBFGS
from ase.optimize.precon import PreconFIRE, PreconLBFGS, PreconODE12r
from ase.cluster import Icosahedron
from ase.build import bulk
def atoms_no_pbc():
    ref_atoms = Icosahedron('Ag', 2, 3.82975)
    ref_atoms.calc = EMT()
    atoms = ref_atoms.copy()
    atoms.calc = EMT()
    atoms.rattle(stdev=0.1, seed=7)
    e_unopt = atoms.get_potential_energy()
    assert e_unopt > 7
    return (atoms, ref_atoms)