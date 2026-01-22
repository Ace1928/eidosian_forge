import difflib
import numpy as np
import os
import re
import glob
import shutil
import sys
import json
import time
import tempfile
import warnings
import subprocess
from copy import deepcopy
from collections import namedtuple
from itertools import product
from typing import List, Set
import ase
import ase.units as units
from ase.calculators.general import Calculator
from ase.calculators.calculator import compare_atoms
from ase.calculators.calculator import PropertyNotImplementedError
from ase.calculators.calculator import kpts2sizeandoffsets
from ase.dft.kpoints import BandPath
from ase.parallel import paropen
from ase.io.castep import read_param
from ase.io.castep import read_bands
from ase.constraints import FixCartesian
def set_pspot(self, pspot, elems=None, notelems=None, clear=True, suffix='usp'):
    """Quickly set all pseudo-potentials: Usually CASTEP psp are named
        like <Elem>_<pspot>.<suffix> so this function function only expects
        the <LibraryName>. It then clears any previous pseudopotential
        settings apply the one with <LibraryName> for each element in the
        atoms object. The optional elems and notelems arguments can be used
        to exclusively assign to some species, or to exclude with notelemens.

        Parameters ::

            - elems (None) : set only these elements
            - notelems (None): do not set the elements
            - clear (True): clear previous settings
            - suffix (usp): PP file suffix
        """
    if self._find_pspots:
        if self._pedantic:
            warnings.warn('Warning: <_find_pspots> = True. Do you really want to use `set_pspots()`? This does not check whether the PP files exist. You may rather want to use `find_pspots()` with the same <pspot>.')
    if clear and (not elems) and (not notelems):
        self.cell.species_pot.clear()
    for elem in set(self.atoms.get_chemical_symbols()):
        if elems is not None and elem not in elems:
            continue
        if notelems is not None and elem in notelems:
            continue
        self.cell.species_pot = (elem, '%s_%s.%s' % (elem, pspot, suffix))