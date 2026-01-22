from operator import itemgetter
from collections import Counter
from itertools import permutations
import numpy as np
from ase.ga.offspring_creator import OffspringCreator
from ase.ga.element_mutations import get_periodic_table_distance
from ase.utils import atoms_to_spglib_cell
def get_minority_element(atoms):
    counter = Counter(atoms.get_chemical_symbols())
    return sorted(counter.items(), key=itemgetter(1), reverse=False)[0][0]