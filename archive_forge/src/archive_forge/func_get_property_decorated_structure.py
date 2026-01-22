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
def get_property_decorated_structure(self):
    """Takes CHGCAR's structure object and updates it with properties
        from the Chargemol analysis.

        Returns:
            Pymatgen structure with site properties added
        """
    struct = self.structure.copy()
    struct.add_site_property('partial_charge_ddec6', self.ddec_charges)
    if self.dipoles:
        struct.add_site_property('dipole_ddec6', self.dipoles)
    if self.bond_order_sums:
        struct.add_site_property('bond_order_sum_ddec6', self.bond_order_sums)
    if self.ddec_spin_moments:
        struct.add_site_property('spin_moment_ddec6', self.ddec_spin_moments)
    if self.cm5_charges:
        struct.add_site_property('partial_charge_cm5', self.cm5_charges)
    return struct