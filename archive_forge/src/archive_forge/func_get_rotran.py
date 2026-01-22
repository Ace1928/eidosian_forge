import numpy as np
from Bio.PDB.PDBExceptions import PDBException
def get_rotran(self):
    """Return right multiplying rotation matrix and translation vector."""
    if self.rot is None:
        raise PDBException('Nothing is superimposed yet.')
    return (self.rot, self.tran)