import numpy as np
import pytest
from ase.cluster.decahedron import Decahedron
from ase.cluster.icosahedron import Icosahedron
from ase.cluster.octahedron import Octahedron
from ase.neighborlist import neighbor_list
@pytest.mark.parametrize('shells', range(1, 7))
def test_cuboctahedron(shells):
    cutoff = shells - 1
    length = 2 * cutoff + 1
    cubocta = Octahedron(sym, length=length, cutoff=cutoff)
    print(cubocta)
    assert len(cubocta) == ico_cubocta_sizes[shells]
    coordination = coordination_numbers(cubocta)
    expected_internal_atoms = ico_cubocta_sizes[shells - 1]
    assert sum(coordination == fcc_maxcoordination) == expected_internal_atoms