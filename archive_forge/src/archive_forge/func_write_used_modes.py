import inspect
import json
import numpy as np
from ase.data import covalent_radii
from ase.neighborlist import NeighborList
from ase.ga.offspring_creator import OffspringCreator
from ase.ga.utilities import atoms_too_close, gather_atoms_by_tag
from scipy.spatial.distance import cdist
def write_used_modes(self, filename):
    """Dump used modes to json file."""
    with open(filename, 'w') as fd:
        json.dump(self.used_modes, fd)
    return