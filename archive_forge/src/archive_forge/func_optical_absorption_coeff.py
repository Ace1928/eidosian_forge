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
@property
def optical_absorption_coeff(self) -> list[float]:
    """
        Calculate the optical absorption coefficient
        from the dielectric constants. Note that this method is only
        implemented for optical properties calculated with GGA and BSE.

        Returns:
            list[float]: optical absorption coefficient
        """
    if self.dielectric_data['density']:
        real_avg = [sum(self.dielectric_data['density'][1][i][0:3]) / 3 for i in range(len(self.dielectric_data['density'][0]))]
        imag_avg = [sum(self.dielectric_data['density'][2][i][0:3]) / 3 for i in range(len(self.dielectric_data['density'][0]))]

        def optical_absorb_coeff(freq, real, imag):
            """
                The optical absorption coefficient calculated in terms of
                equation, the unit is cm^-1.
                """
            hc = 1.23984 * 0.0001
            return 2 * 3.14159 * np.sqrt(np.sqrt(real ** 2 + imag ** 2) - real) * np.sqrt(2) / hc * freq
        absorption_coeff = list(itertools.starmap(optical_absorb_coeff, zip(self.dielectric_data['density'][0], real_avg, imag_avg)))
    return absorption_coeff