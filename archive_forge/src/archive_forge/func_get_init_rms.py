import numpy as np
from Bio.PDB.PDBExceptions import PDBException
def get_init_rms(self):
    """Return the root mean square deviation of untransformed coordinates."""
    if self.coords is None:
        raise PDBException('No coordinates set yet.')
    if self.init_rms is None:
        diff = self.coords - self.reference_coords
        self.init_rms = np.sqrt(np.sum(np.sum(diff * diff, axis=1) / self._natoms))
    return self.init_rms