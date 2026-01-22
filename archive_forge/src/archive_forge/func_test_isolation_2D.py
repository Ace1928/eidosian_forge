import pytest
import ase.build
from ase import Atoms
from ase.lattice.cubic import FaceCenteredCubic
from ase.geometry.dimensionality import (analyze_dimensionality,
def test_isolation_2D():
    atoms = ase.build.mx2(formula='MoS2', kind='2H', a=3.18, thickness=3.19)
    atoms.cell[2, 2] = 7
    atoms.set_pbc((1, 1, 1))
    atoms *= 2
    result = isolate_components(atoms)
    assert len(result) == 1
    key, components = list(result.items())[0]
    assert key == '2D'
    assert len(components) == 2
    for layer in components:
        empirical = atoms.get_chemical_formula(empirical=True)
        assert empirical == layer.get_chemical_formula(empirical=True)
        assert (layer.pbc == [True, True, False]).all()