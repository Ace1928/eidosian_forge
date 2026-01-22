from __future__ import annotations
import os
import subprocess
import warnings
from glob import glob
from shutil import which
from typing import TYPE_CHECKING
import numpy as np
from monty.tempfile import ScratchDir
from pymatgen.core import Element
from pymatgen.io.vasp.inputs import Potcar
from pymatgen.io.vasp.outputs import Chgcar
def get_bond_order(self, index_from, index_to):
    """Convenience method to get the bond order between two atoms.

        Args:
            index_from (int): Index of atom to get bond order from.
            index_to (int): Index of atom to get bond order to.

        Returns:
            float: bond order between atoms
        """
    bonded_set = self.bond_order_dict[index_from]['bonded_to']
    bond_orders = [v['bond_order'] for v in bonded_set if v['index'] == index_to]
    return 0.0 if bond_orders == [] else np.sum(bond_orders)