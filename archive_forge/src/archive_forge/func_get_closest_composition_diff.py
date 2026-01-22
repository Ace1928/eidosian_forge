from operator import itemgetter
from collections import Counter
from itertools import permutations
import numpy as np
from ase.ga.offspring_creator import OffspringCreator
from ase.ga.element_mutations import get_periodic_table_distance
from ase.utils import atoms_to_spglib_cell
def get_closest_composition_diff(self, c):
    comp = np.array(c)
    mindiff = 10000000000.0
    allowed_list = list(self.allowed_compositions)
    self.rng.shuffle(allowed_list)
    for ac in allowed_list:
        diff = self.get_composition_diff(comp, ac)
        numdiff = sum([abs(i) for i in diff])
        if numdiff < mindiff:
            mindiff = numdiff
            ccdiff = diff
    return ccdiff