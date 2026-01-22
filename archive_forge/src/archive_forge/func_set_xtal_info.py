from Bio.PDB.StructureBuilder import StructureBuilder
import numpy as np
def set_xtal_info(self, space_group, unit_cell):
    """Set the crystallographic information for the structure.

        :param space_group: the space group name, e.g. "P 21 21 21"
        :param unit_cell: an array of length 6 with the unit cell parameters in order: a, b, c, alpha, beta, gamma

        """
    self.structure_builder.set_symmetry(space_group, unit_cell)