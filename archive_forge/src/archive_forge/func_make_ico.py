import numpy as np
from ase.cluster import Icosahedron
from ase.ga.particle_comparator import NNMatComparator
from ase.ga.utilities import get_nnmat
from ase.ga.particle_mutations import RandomPermutation
def make_ico(sym):
    atoms = Icosahedron(sym, 4)
    atoms.center(vacuum=4.0)
    return atoms