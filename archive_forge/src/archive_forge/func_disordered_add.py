import copy
import sys
import warnings
import numpy as np
from Bio.PDB.Entity import DisorderedEntityWrapper
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from Bio.PDB.vectors import Vector
from Bio.Data import IUPACData
def disordered_add(self, atom):
    """Add a disordered atom."""
    atom.flag_disorder()
    residue = self.get_parent()
    atom.set_parent(residue)
    altloc = atom.get_altloc()
    occupancy = atom.get_occupancy()
    self[altloc] = atom
    if occupancy > self.last_occupancy:
        self.last_occupancy = occupancy
        self.disordered_select(altloc)