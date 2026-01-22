from __future__ import annotations
import re
from typing import TYPE_CHECKING, no_type_check
import numpy as np
from monty.io import zopen
from pymatgen.core.structure import Structure
from pymatgen.core.units import Ry_to_eV, bohr_to_angstrom
from pymatgen.electronic_structure.core import Spin
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.util.num import round_to_sigfigs

        Subroutine to extract bond label, site indices, and length from
        a COPL header line. The site indices are zero-based, so they
        can be easily used with a Structure object.

        Example header line: Fe-1/Fe-1-tr(-1,-1,-1) : 2.482 Ang.

        Args:
            line: line in the COHPCAR header describing the bond.

        Returns:
            The bond label, the bond length and a tuple of the site indices.
        