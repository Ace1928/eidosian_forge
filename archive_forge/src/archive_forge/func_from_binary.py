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
@classmethod
def from_binary(cls, filename: str, data_type: str='complex64') -> Self:
    """Read the WAVEDER file and returns a Waveder object.

        Args:
            filename: Name of file containing WAVEDER.
            data_type: Data type of the WAVEDER file. Default is complex64.
                If the file was generated with the "gamma" version of VASP,
                the data type can be either "float64" or "float32".

        Returns:
            Waveder object.
        """
    with open(filename, 'rb') as file:

        def read_data(dtype):
            """Read records from Fortran binary file and convert to np.array of given dtype."""
            data = b''
            while True:
                prefix = np.fromfile(file, dtype=np.int32, count=1)[0]
                data += file.read(abs(prefix))
                suffix = np.fromfile(file, dtype=np.int32, count=1)[0]
                if abs(prefix) - abs(suffix):
                    raise RuntimeError(f'Read wrong amount of bytes.\nExpected: {prefix}, read: {len(data)}, suffix: {suffix}.')
                if prefix > 0:
                    break
            return np.frombuffer(data, dtype=dtype)
        nbands, nelect, nk, ispin = read_data(np.int32)
        _ = read_data(np.float64)
        _ = read_data(np.float64)
        me_datatype = np.dtype(data_type)
        cder = read_data(me_datatype)
        cder_data = cder.reshape((3, ispin, nk, nelect, nbands)).T
        return cls(cder_data.real, cder_data.imag)