import inspect
import json
import numpy as np
from ase.data import covalent_radii
from ase.neighborlist import NeighborList
from ase.ga.offspring_creator import OffspringCreator
from ase.ga.utilities import atoms_too_close, gather_atoms_by_tag
from scipy.spatial.distance import cdist
def get_new_individual(self, parents):
    f = parents[0]
    indi = self.mutate(f)
    if indi is None:
        return (indi, 'mutation: soft')
    indi = self.initialize_individual(f, indi)
    indi.info['data']['parents'] = [f.info['confid']]
    return (self.finalize_individual(indi), 'mutation: soft')