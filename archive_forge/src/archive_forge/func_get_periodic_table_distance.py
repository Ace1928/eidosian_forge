import numpy as np
from ase.data import atomic_numbers
from ase.ga.offspring_creator import OffspringCreator
def get_periodic_table_distance(e1, e2):
    rc1 = np.array(get_row_column(e1))
    rc2 = np.array(get_row_column(e2))
    return sum(np.abs(rc1 - rc2))