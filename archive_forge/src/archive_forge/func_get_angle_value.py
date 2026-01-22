from ase.neighborlist import build_neighbor_list, get_distance_matrix, get_distance_indices
from ase.ga.utilities import get_rdf
from ase import Atoms
def get_angle_value(self, imIdx, idxs, mic=True, **kwargs):
    """Get angle.

        Parameters:

        imIdx: int
            Index of Image to get value from.
        idxs: tuple or list of integers
            Get angle between atoms idxs[0]-idxs[1]-idxs[2].
        mic: bool
            Passed on to :func:`ase.Atoms.get_angle` for retrieving the value, defaults to True.
            If the cell of the image is correctly set, there should be no reason to change this.
        kwargs: options or dict
            Passed on to :func:`ase.Atoms.get_angle`.

        Returns:

        return: float
            Value returned by image.get_angle.
        """
    return self.images[imIdx].get_angle(idxs[0], idxs[1], idxs[2], mic=True, **kwargs)