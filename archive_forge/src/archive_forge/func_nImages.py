from ase.neighborlist import build_neighbor_list, get_distance_matrix, get_distance_indices
from ase.ga.utilities import get_rdf
from ase import Atoms
@property
def nImages(self):
    """Number of Images in this instance.

        Cannot be set, is determined automatically.
        """
    return len(self.images)