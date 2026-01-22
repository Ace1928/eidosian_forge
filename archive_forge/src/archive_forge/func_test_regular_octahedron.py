import numpy as np
import pytest
from ase.cluster.decahedron import Decahedron
from ase.cluster.icosahedron import Icosahedron
from ase.cluster.octahedron import Octahedron
from ase.neighborlist import neighbor_list
@pytest.mark.parametrize('shells', range(1, 8))
def test_regular_octahedron(shells):
    octa = Octahedron(sym, length=shells, cutoff=0)
    coordination = coordination_numbers(octa)
    assert len(octa) == octa_sizes[shells]
    if shells == 1:
        return
    assert min(coordination) == 4
    assert sum(coordination == 4) == 6
    expected_internal_atoms = octa_sizes[shells - 2]
    assert sum(coordination == fcc_maxcoordination) == expected_internal_atoms