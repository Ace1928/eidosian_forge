from operator import itemgetter
from collections import Counter
from itertools import permutations
import numpy as np
from ase.ga.offspring_creator import OffspringCreator
from ase.ga.element_mutations import get_periodic_table_distance
from ase.utils import atoms_to_spglib_cell
def minority_element_segregate(atoms, layer_tag=1, rng=np.random):
    """Move the minority alloy element to the layer specified by the layer_tag,
    Atoms object should contain atoms with the corresponding tag."""
    sym = get_minority_element(atoms)
    layer_indices = set([a.index for a in atoms if a.tag == layer_tag])
    minority_indices = set([a.index for a in atoms if a.symbol == sym])
    change_indices = minority_indices - layer_indices
    in_layer_not_sym = list(layer_indices - minority_indices)
    rng.shuffle(in_layer_not_sym)
    if len(change_indices) > 0:
        for i, ai in zip(change_indices, in_layer_not_sym):
            atoms[i].symbol = atoms[ai].symbol
            atoms[ai].symbol = sym