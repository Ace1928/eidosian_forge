from operator import itemgetter
from collections import Counter
from itertools import permutations
import numpy as np
from ase.ga.offspring_creator import OffspringCreator
from ase.ga.element_mutations import get_periodic_table_distance
from ase.utils import atoms_to_spglib_cell
def replace_element(atoms, element_out, element_in):
    syms = np.array(atoms.get_chemical_symbols())
    syms[syms == element_out] = element_in
    atoms.set_chemical_symbols(syms)