from ase.neighborlist import build_neighbor_list, get_distance_matrix, get_distance_indices
from ase.ga.utilities import get_rdf
from ase import Atoms
@property
def unique_dihedrals(self):
    """Get Unique Dihedrals.

        :data:`all_dihedrals` i-j-k-l without l-k-j-i.

        """
    return self._filter_unique(self.all_dihedrals)