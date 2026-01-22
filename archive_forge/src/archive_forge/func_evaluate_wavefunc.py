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
def evaluate_wavefunc(self, kpoint: int, band: int, r: np.ndarray, spin: int=0, spinor: int=0) -> np.complex64:
    """
        Evaluates the wavefunction for a given position, r.

        The wavefunction is given by the k-point and band. It is evaluated
        at the given position by summing over the components. Formally,

        \\psi_n^k (r) = \\sum_{i=1}^N c_i^{n,k} \\exp (i (k + G_i^{n,k}) \\cdot r)

        where \\psi_n^k is the wavefunction for the nth band at k-point k, N is
        the number of plane waves, c_i^{n,k} is the ith coefficient that
        corresponds to the nth band and k-point k, and G_i^{n,k} is the ith
        G-point corresponding to k-point k.

        NOTE: This function is very slow; a discrete fourier transform is the
        preferred method of evaluation (see Wavecar.fft_mesh).

        Args:
            kpoint (int): the index of the kpoint where the wavefunction will be evaluated
            band (int): the index of the band where the wavefunction will be evaluated
            r (np.array): the position where the wavefunction will be evaluated
            spin (int): spin index for the desired wavefunction (only for
                ISPIN = 2, default = 0)
            spinor (int): component of the spinor that is evaluated (only used
                if vasp_type == 'ncl')

        Returns:
            a complex value corresponding to the evaluation of the wavefunction
        """
    v = self.Gpoints[kpoint] + self.kpoints[kpoint]
    u = np.dot(np.dot(v, self.b), r)
    if self.vasp_type.lower()[0] == 'n':
        c = self.coeffs[kpoint][band][spinor, :]
    elif self.spin == 2:
        c = self.coeffs[spin][kpoint][band]
    else:
        c = self.coeffs[kpoint][band]
    return np.sum(np.dot(c, np.exp(1j * u, dtype=np.complex64))) / np.sqrt(self.vol)