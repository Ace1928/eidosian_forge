import numpy as np
from ase.data import atomic_numbers
from ase.ga.offspring_creator import OffspringCreator
def get_mutation_index_list_and_choices(self, atoms):
    """Returns a list of the indices that are going to
        be mutated and a list of possible elements to mutate
        to. The lists obey the criteria set in the initialization.
        """
    itbm_ok = False
    while not itbm_ok:
        itbm = self.rng.choice(range(len(atoms)))
        itbm_ok = True
        for i, e in enumerate(self.element_pools):
            if atoms[itbm].symbol in e:
                elems = e[:]
                elems_in, indices_in = zip(*[(a.symbol, a.index) for a in atoms if a.symbol in elems])
                max_diff_elem = self.max_diff_elements[i]
                min_percent_elem = self.min_percentage_elements[i]
                if min_percent_elem == 0:
                    min_percent_elem = 1.0 / len(elems_in)
                break
        else:
            itbm_ok = False
    diff_elems_in = len(set(elems_in))
    if diff_elems_in == max_diff_elem:
        ltbm = []
        for i in range(len(atoms)):
            if atoms[i].symbol == atoms[itbm].symbol:
                ltbm.append(i)
    else:
        if self.verbose:
            print(int(min_percent_elem * len(elems_in)), min_percent_elem, len(elems_in))
        all_chunks = chunks(indices_in, int(min_percent_elem * len(elems_in)))
        itbm_num_of_elems = 0
        for a in atoms:
            if a.index == itbm:
                break
            if a.symbol in elems:
                itbm_num_of_elems += 1
        ltbm = all_chunks[itbm_num_of_elems // int(min_percent_elem * len(elems_in)) - 1]
    elems.remove(atoms[itbm].symbol)
    return (ltbm, elems)