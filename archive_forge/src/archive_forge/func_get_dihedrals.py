from ase.neighborlist import build_neighbor_list, get_distance_matrix, get_distance_indices
from ase.ga.utilities import get_rdf
from ase import Atoms
def get_dihedrals(self, A, B, C, D, unique=True):
    """Get dihedrals A-B-C-D.

        Parameters:

        A, B, C, D: str
            Get Dihedralss between elements A, B, C and D. **B-C will be the central axis**.
        unique: bool
            Return the dihedrals both ways or just one way (A-B-C-D and D-C-B-A or only A-B-C-D)

        Returns:

        return: list of lists of tuples
            return[imageIdx][atomIdx][dihedralI], each tuple starts with atomIdx.

        Use :func:`get_values` to convert the returned list to values.
        """
    r = []
    for imI in range(len(self.all_dihedrals)):
        r.append([])
        aIdxs = self._get_symbol_idxs(imI, A)
        bIdxs = self._get_symbol_idxs(imI, B)
        cIdxs = self._get_symbol_idxs(imI, C)
        dIdxs = self._get_symbol_idxs(imI, D)
        for aIdx in aIdxs:
            dihedrals = [(aIdx,) + d for d in self.all_dihedrals[imI][aIdx] if d[0] in bIdxs and d[1] in cIdxs and (d[2] in dIdxs)]
            if not unique:
                dihedrals += [d[::-1] for d in dihedrals]
            r[-1].extend(dihedrals)
    return r