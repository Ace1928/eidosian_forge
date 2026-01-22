from operator import itemgetter
from collections import Counter
from itertools import permutations
import numpy as np
from ase.ga.offspring_creator import OffspringCreator
from ase.ga.element_mutations import get_periodic_table_distance
from ase.utils import atoms_to_spglib_cell
def get_add_remove_elements(self, syms):
    if self.element_pools is None or self.allowed_compositions is None:
        return ([], [])
    unique_syms, pool_number, comp = get_ordered_composition(syms, self.element_pools)
    stay_comp, stay_syms = ([], [])
    add_rem = {}
    per_pool = len(self.allowed_compositions[0]) / len(self.element_pools)
    pool_count = np.zeros(len(self.element_pools), dtype=int)
    for pn, num, sym in zip(pool_number, comp, unique_syms):
        pool_count[pn] += 1
        if pool_count[pn] <= per_pool:
            stay_comp.append(num)
            stay_syms.append(sym)
        else:
            add_rem[sym] = -num
    diff = self.get_closest_composition_diff(stay_comp)
    add_rem.update(dict(((s, c) for s, c in zip(stay_syms, diff))))
    return get_add_remove_lists(**add_rem)