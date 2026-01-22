import numpy as np
from operator import itemgetter
from ase.ga.offspring_creator import OffspringCreator
from ase.ga.utilities import get_distance_matrix, get_nndist
from ase import Atoms
@classmethod
def get_list_of_possible_permutations(cls, atoms, l1, l2):
    """Returns a list of available permutations from the two
        lists of indices, l1 and l2. Checking that identical elements
        are not permuted."""
    possible_permutations = []
    for i in l1:
        for j in l2:
            if atoms[int(i)].symbol != atoms[int(j)].symbol:
                possible_permutations.append((i, j))
    return possible_permutations