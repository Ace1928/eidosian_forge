import os
import sys
import re
import numpy as np
import subprocess
from contextlib import contextmanager
from pathlib import Path
from warnings import warn
from typing import Dict, Any
from xml.etree import ElementTree
import ase
from ase.io import read, jsonio
from ase.utils import PurePath
from ase.calculators import calculator
from ase.calculators.calculator import Calculator
from ase.calculators.singlepoint import SinglePointDFTCalculator
from ase.calculators.vasp.create_input import GenerateVaspInput
def update_atoms(self, atoms):
    """Update the atoms object with new positions and cell"""
    if self.int_params['ibrion'] is not None and self.int_params['nsw'] is not None:
        if self.int_params['ibrion'] > -1 and self.int_params['nsw'] > 0:
            atoms_sorted = read(self._indir('CONTCAR'))
            atoms.positions = atoms_sorted[self.resort].positions
            atoms.cell = atoms_sorted.cell
    self.atoms = atoms