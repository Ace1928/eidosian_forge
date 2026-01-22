from collections import defaultdict
import numpy as np
import kimpy
from kimpy import neighlist
from ase.neighborlist import neighbor_list
from ase import Atom
from .kimpy_wrappers import check_call_wrapper
@staticmethod
def get_neigh(data, cutoffs, neighbor_list_index, particle_number):
    """Retrieves the neighbors of each atom using ASE's native neighbor
        list library
        """
    number_of_particles = data['num_particles']
    if particle_number >= number_of_particles or particle_number < 0:
        return (np.array([]), 1)
    neighbors = data['neighbors'][neighbor_list_index][particle_number]
    return (neighbors, 0)