from __future__ import annotations
import collections
import fnmatch
import os
import re
import warnings
from collections import defaultdict
from typing import TYPE_CHECKING, Any
import numpy as np
from monty.io import zopen
from monty.json import MSONable
from pymatgen.core.structure import Structure
from pymatgen.electronic_structure.bandstructure import LobsterBandStructureSymmLine
from pymatgen.electronic_structure.core import Orbital, Spin
from pymatgen.electronic_structure.dos import Dos, LobsterCompleteDos
from pymatgen.io.vasp.inputs import Kpoints
from pymatgen.io.vasp.outputs import Vasprun, VolumetricData
from pymatgen.util.due import Doi, due
def get_structure_with_site_potentials(self, structure_filename):
    """
        Get a Structure with Mulliken and Loewdin charges as site properties

        Args:
            structure_filename: filename of POSCAR

        Returns:
            Structure Object with Mulliken and Loewdin charges as site properties.
        """
    struct = Structure.from_file(structure_filename)
    mulliken = self.sitepotentials_mulliken
    loewdin = self.sitepotentials_loewdin
    site_properties = {'Mulliken Site Potentials (eV)': mulliken, 'Loewdin Site Potentials (eV)': loewdin}
    return struct.copy(site_properties=site_properties)