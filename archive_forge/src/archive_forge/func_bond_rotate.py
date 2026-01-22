import re
from collections import deque, namedtuple
import copy
from numbers import Integral
import numpy as np  # type: ignore
from Bio.PDB.Atom import Atom, DisorderedAtom
from Bio.Data.PDBData import protein_letters_3to1
from Bio.PDB.vectors import multi_coord_space, multi_rot_Z
from Bio.PDB.vectors import coord_space
from Bio.PDB.ic_data import ic_data_backbone, ic_data_sidechains
from Bio.PDB.ic_data import primary_angles
from Bio.PDB.ic_data import ic_data_sidechain_extras, residue_atom_bond_state
from Bio.PDB.ic_data import dihedra_primary_defaults, hedra_defaults
from typing import (
def bond_rotate(self, angle_key: Union[EKT, str], delta: float):
    """Rotate set of overlapping dihedrals by delta degrees.

        Changes a dihedral angle by a given delta, i.e.
        new_angle = current_angle + delta
        Values are adjusted so new_angle iwll be within +/-180.

        Changes overlapping dihedra as in :meth:`.set_angle`

        See :meth:`.pick_angle` for key specifications.
        """
    base = self.pick_angle(angle_key)
    if base is not None:
        self._do_bond_rotate(base, delta)