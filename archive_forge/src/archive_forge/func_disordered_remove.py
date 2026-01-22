import copy
import sys
import warnings
import numpy as np
from Bio.PDB.Entity import DisorderedEntityWrapper
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from Bio.PDB.vectors import Vector
from Bio.Data import IUPACData
def disordered_remove(self, altloc):
    """Remove a child atom altloc from the DisorderedAtom.

        Arguments:
         - altloc - name of the altloc to remove, as a string.

        """
    atom = self.child_dict[altloc]
    is_selected = self.selected_child is atom
    del self.child_dict[altloc]
    atom.detach_parent()
    if is_selected and self.child_dict:
        child = sorted(self.child_dict.values(), key=lambda a: a.occupancy)[-1]
        self.disordered_select(child.altloc)
    elif not self.child_dict:
        self.selected_child = None
        self.last_occupancy = -sys.maxsize