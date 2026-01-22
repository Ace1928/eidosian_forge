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
def distplot_to_dh_arrays(self, distplot: np.ndarray, dihedra_signs: np.ndarray) -> None:
    """Load di/hedra distance arays from distplot.

        Fill :class:`IC_Chain` arrays hedraL12, L23, L13 and dihedraL14
        distance value arrays from input distplot, dihedra_signs array from
        input dihedra_signs.  Distplot and di/hedra distance arrays must index
        according to AtomKey mappings in :class:`IC_Chain` .hedraNdx and .dihedraNdx
        (created in :meth:`IC_Chain.init_edra`)

        Call :meth:`atom_to_internal_coordinates` (or at least :meth:`init_edra`)
        to generate a2ha_map and d2a_map before running this.

        Explcitly removed from :meth:`.distance_to_internal_coordinates` so
        user may populate these chain di/hedra arrays by other
        methods.
        """
    ha = self.a2ha_map.reshape(-1, 3)
    self.hedraL12 = distplot[ha[:, 0], ha[:, 1]]
    self.hedraL23 = distplot[ha[:, 1], ha[:, 2]]
    self.hedraL13 = distplot[ha[:, 0], ha[:, 2]]
    da = self.d2a_map
    self.dihedraL14 = distplot[da[:, 0], da[:, 3]]
    self.dihedra_signs = dihedra_signs