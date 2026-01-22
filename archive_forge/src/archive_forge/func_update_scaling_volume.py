import numpy as np
from ase import Atoms
from ase.geometry import find_mic
from ase.ga.utilities import (atoms_too_close, atoms_too_close_two_sets,
from ase.ga.offspring_creator import OffspringCreator
def update_scaling_volume(self, population, w_adapt=0.5, n_adapt=0):
    """Updates the scaling volume that is used in the pairing

        w_adapt: weight of the new vs the old scaling volume
        n_adapt: number of best candidates in the population that
                 are used to calculate the new scaling volume
        """
    if not n_adapt:
        n_adapt = int(np.ceil(0.2 * len(population)))
    v_new = np.mean([a.get_volume() for a in population[:n_adapt]])
    if not self.scaling_volume:
        self.scaling_volume = v_new
    else:
        volumes = [self.scaling_volume, v_new]
        weights = [1 - w_adapt, w_adapt]
        self.scaling_volume = np.average(volumes, weights=weights)