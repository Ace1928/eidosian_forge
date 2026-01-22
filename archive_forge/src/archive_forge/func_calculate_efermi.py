from __future__ import annotations
import datetime
import itertools
import logging
import math
import os
import re
import warnings
import xml.etree.ElementTree as ET
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from glob import glob
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING, Literal
import numpy as np
from monty.io import reverse_readfile, zopen
from monty.json import MSONable, jsanitize
from monty.os.path import zpath
from monty.re import regrep
from numpy.testing import assert_allclose
from pymatgen.core import Composition, Element, Lattice, Structure
from pymatgen.core.units import unitized
from pymatgen.electronic_structure.bandstructure import (
from pymatgen.electronic_structure.core import Magmom, Orbital, OrbitalType, Spin
from pymatgen.electronic_structure.dos import CompleteDos, Dos
from pymatgen.entries.computed_entries import ComputedEntry, ComputedStructureEntry
from pymatgen.io.common import VolumetricData as BaseVolumetricData
from pymatgen.io.core import ParseError
from pymatgen.io.vasp.inputs import Incar, Kpoints, Poscar, Potcar
from pymatgen.io.wannier90 import Unk
from pymatgen.util.io_utils import clean_lines, micro_pyawk
from pymatgen.util.num import make_symmetric_matrix_from_upper_tri
def calculate_efermi(self, tol: float=0.001):
    """
        Calculate the Fermi level using a robust algorithm.

        Sometimes VASP can put the Fermi level just inside of a band due to issues in
        the way band occupancies are handled. This algorithm tries to detect and correct
        for this bug.

        Slightly more details are provided here: https://www.vasp.at/forum/viewtopic.php?f=4&t=17981
        """
    all_eigs = np.concatenate([eigs[:, :, 0].transpose(1, 0) for eigs in self.eigenvalues.values()])

    def crosses_band(fermi):
        eigs_below = np.any(all_eigs < fermi, axis=1)
        eigs_above = np.any(all_eigs > fermi, axis=1)
        return np.any(eigs_above & eigs_below)

    def get_vbm_cbm(fermi):
        return (np.max(all_eigs[all_eigs < fermi]), np.min(all_eigs[all_eigs > fermi]))
    if not crosses_band(self.efermi):
        return self.efermi
    if not crosses_band(self.efermi + tol):
        vbm, cbm = get_vbm_cbm(self.efermi + tol)
        return (cbm + vbm) / 2
    if not crosses_band(self.efermi - tol):
        vbm, cbm = get_vbm_cbm(self.efermi - tol)
        return (cbm + vbm) / 2
    return self.efermi