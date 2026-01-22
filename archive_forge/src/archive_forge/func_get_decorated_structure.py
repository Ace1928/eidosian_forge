from __future__ import annotations
import os
import shutil
import subprocess
import warnings
from datetime import datetime
from glob import glob
from pathlib import Path
from shutil import which
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any
import numpy as np
from monty.dev import deprecated
from monty.shutil import decompress_file
from monty.tempfile import ScratchDir
from pymatgen.io.common import VolumetricData
from pymatgen.io.vasp.inputs import Potcar
from pymatgen.io.vasp.outputs import Chgcar
def get_decorated_structure(self, property_name: str, average: bool=False) -> Structure:
    """Get a property-decorated structure from the Bader analysis.

        This is distinct from getting charge decorated structure, which assumes
        the "standard" Bader analysis of electron densities followed by converting
        electron count to charge. The expected way to use this is to call Bader on
        a non-charge density file such as a spin density file, electrostatic potential
        file, etc., while using the charge density file as the reference (chgref_filename)
        so that the partitioning is determined via the charge, but averaging or integrating
        is done for another property.

        User warning: Bader analysis cannot automatically determine what property is
        inside of the file. So if you want to use this for a non-conventional property
        like spin, you must ensure that you have the file is for the appropriate
        property and you have an appropriate reference file.

        Args:
            property_name (str): name of the property to assign to the structure, note that
                if name is "spin" this is handled as a special case, and the appropriate
                spin properties are set on the species in the structure
            average (bool): whether or not to return the average of this property, rather
                than the total, by dividing by the atomic volume.

        Returns:
            structure with site properties assigned via Bader Analysis
        """
    vals = np.array([self.get_charge(i) for i in range(len(self.structure))])
    struct = self.structure.copy()
    if average:
        vals = np.divide(vals, [d['atomic_vol'] for d in self.data])
    struct.add_site_property(property_name, vals)
    if property_name == 'spin':
        struct.add_spin_by_site(vals)
    return struct