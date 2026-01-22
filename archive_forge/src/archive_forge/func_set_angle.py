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
def set_angle(self, angle_key: Union[EKT, str], v: float, overlap=True):
    """Set dihedron or hedron angle for specified key.

        If angle is a `Dihedron` and `overlap` is True (default), overlapping
        dihedra are also changed as appropriate.  The overlap is a result of
        protein chain definitions in :mod:`.ic_data` and :meth:`_create_edra`
        (e.g. psi overlaps N-CA-C-O).

        Te default overlap=True is probably what you want for:
        `set_angle("chi1", val)`

        The default is probably NOT what you want when processing all dihedrals
        in a chain or residue (such as copying from another structure), as the
        overlaping dihedra will likely be in the set as well.

        N.B. setting e.g. PRO chi2 is permitted without error or warning!

        See :meth:`.pick_angle` for angle_key specifications.
        See :meth:`.bond_rotate` to change a dihedral by a number of degrees

        :param angle_key: angle identifier.
        :param float v: new angle in degrees (result adjusted to +/-180).
        :param bool overlap: default True.
            Modify overlapping dihedra as needed
        """
    edron = self.pick_angle(angle_key)
    if edron is None:
        return
    elif isinstance(edron, Hedron) or not overlap:
        edron.angle = v
    else:
        delta = Dihedron.angle_dif(edron.angle, v)
        self._do_bond_rotate(edron, delta)