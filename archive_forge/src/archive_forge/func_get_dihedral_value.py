from ase.neighborlist import build_neighbor_list, get_distance_matrix, get_distance_indices
from ase.ga.utilities import get_rdf
from ase import Atoms
def get_dihedral_value(self, imIdx, idxs, mic=True, **kwargs):
    """Get dihedral.

        Parameters:

        imIdx: int
            Index of Image to get value from.
        idxs: tuple or list of integers
            Get angle between atoms idxs[0]-idxs[1]-idxs[2]-idxs[3].
        mic: bool
            Passed on to :func:`ase.Atoms.get_dihedral` for retrieving the value, defaults to True.
            If the cell of the image is correctly set, there should be no reason to change this.
        kwargs: options or dict
            Passed on to :func:`ase.Atoms.get_dihedral`.

        Returns:

        return: float
            Value returned by image.get_dihedral.
        """
    return self.images[imIdx].get_dihedral(idxs[0], idxs[1], idxs[2], idxs[3], mic=mic, **kwargs)