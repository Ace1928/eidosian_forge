import numpy as np
from operator import itemgetter
from ase.ga.offspring_creator import OffspringCreator
from ase.ga.utilities import get_distance_matrix, get_nndist
from ase import Atoms
@classmethod
def random_vector(cls, l, rng=np.random):
    """return random vector of length l"""
    vec = np.array([rng.rand() * 2 - 1 for i in range(3)])
    vl = np.linalg.norm(vec)
    return np.array([v * l / vl for v in vec])