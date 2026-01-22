import numpy as np
import pytest
from ase.cluster.decahedron import Decahedron
from ase.cluster.icosahedron import Icosahedron
from ase.cluster.octahedron import Octahedron
from ase.neighborlist import neighbor_list
def test_smallest_decahedron():
    assert len(Decahedron(sym, 1, 1, 0)) == 1