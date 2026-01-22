import numpy as np
from Bio.PDB.PDBExceptions import PDBException
def get_transformed(self):
    """Get the transformed coordinate set."""
    if self.coords is None or self.reference_coords is None:
        raise PDBException('No coordinates set.')
    if self.rot is None:
        raise PDBException('Nothing is superimposed yet.')
    self.transformed_coords = np.dot(self.coords, self.rot) + self.tran
    return self.transformed_coords