from __future__ import annotations
import os
import re
import warnings
from string import Template
from typing import TYPE_CHECKING
import numpy as np
from monty.io import zopen
from monty.json import MSONable
from pymatgen.analysis.excitation import ExcitationSpectrum
from pymatgen.core.structure import Molecule, Structure
from pymatgen.core.units import Energy, FloatWithUnit
@classmethod
def dft_task(cls, mol, xc='b3lyp', **kwargs):
    """
        A class method for quickly creating DFT tasks with optional
        cosmo parameter .

        Args:
            mol: Input molecule
            xc: Exchange correlation to use.
            kwargs: Any of the other kwargs supported by NwTask. Note the
                theory is always "dft" for a dft task.
        """
    t = NwTask.from_molecule(mol, theory='dft', **kwargs)
    t.theory_directives.update({'xc': xc, 'mult': t.spin_multiplicity})
    return t