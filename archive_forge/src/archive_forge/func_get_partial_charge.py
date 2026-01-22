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
def get_partial_charge(self, atom_index, charge_type='ddec'):
    """Convenience method to get the partial atomic charge on a particular atom.
        This is the value printed in the Chargemol analysis.

        Args:
            atom_index (int): Index of atom to get charge for.
            charge_type (str): Type of charge to use ("ddec" or "cm5").
        """
    if charge_type.lower() not in ['ddec', 'cm5']:
        raise ValueError(f'Invalid charge_type: {charge_type}')
    if charge_type.lower() == 'ddec':
        partial_charge = self.ddec_charges[atom_index]
    elif charge_type.lower() == 'cm5':
        partial_charge = self.cm5_charges[atom_index]
    return partial_charge