import copy
import sys
import warnings
import numpy as np
from Bio.PDB.Entity import DisorderedEntityWrapper
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from Bio.PDB.vectors import Vector
from Bio.Data import IUPACData
def set_anisou(self, anisou_array):
    """Set anisotropic B factor.

        :param anisou_array: anisotropic B factor.
        :type anisou_array: NumPy array (length 6)
        """
    self.anisou_array = anisou_array