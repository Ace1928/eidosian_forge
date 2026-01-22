from __future__ import annotations
import logging
import os
import re
import warnings
from glob import glob
from itertools import chain
import numpy as np
import pandas as pd
from monty.io import zopen
from monty.json import MSONable, jsanitize
from monty.re import regrep
from pymatgen.core.structure import Molecule, Structure
from pymatgen.core.units import Ha_to_eV
from pymatgen.electronic_structure.bandstructure import BandStructure, BandStructureSymmLine
from pymatgen.electronic_structure.core import Orbital, Spin
from pymatgen.electronic_structure.dos import CompleteDos, Dos
from pymatgen.io.cp2k.inputs import Keyword
from pymatgen.io.cp2k.sets import Cp2kInput
from pymatgen.io.cp2k.utils import natural_keys, postprocessor
from pymatgen.io.xyz import XYZ
def parse_energy_file(energy_file):
    """Parses energy file for calculations with multiple ionic steps."""
    columns = ['step', 'kinetic_energy', 'temp', 'potential_energy', 'conserved_quantity', 'used_time']
    df = pd.read_csv(energy_file, skiprows=1, names=columns, sep='\\s+')
    df['kinetic_energy'] = df['kinetic_energy'] * Ha_to_eV
    df['potential_energy'] = df['potential_energy'] * Ha_to_eV
    df['conserved_quantity'] = df['conserved_quantity'] * Ha_to_eV
    df.astype(float)
    return {c: df[c].to_numpy() for c in columns}