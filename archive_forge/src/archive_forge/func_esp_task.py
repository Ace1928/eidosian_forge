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
def esp_task(cls, mol, **kwargs):
    """
        A class method for quickly creating ESP tasks with RESP
        charge fitting.

        Args:
            mol: Input molecule
            kwargs: Any of the other kwargs supported by NwTask. Note the
                theory is always "dft" for a dft task.
        """
    return NwTask.from_molecule(mol, theory='esp', **kwargs)