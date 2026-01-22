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
def find_pspots(self, pspot='.+', elems=None, notelems=None, clear=True, suffix='(usp|UPF|recpot)'):
    """Quickly find and set all pseudo-potentials by searching in
        castep_pp_path:

        This one is more flexible than set_pspots, and also checks if the files
        are actually available from the castep_pp_path.

        Essentially, the function parses the filenames in <castep_pp_path> and
        does a regex matching. The respective pattern is:

            r"^(<elem>|<elem.upper()>|elem.lower()>(_|-)<pspot>\\.<suffix>$"

        In most cases, it will be sufficient to not specify anything, if you
        use standard CASTEP USPPs with only one file per element in the
        <castep_pp_path>.

        The function raises a `RuntimeError` if there is some ambiguity
        (multiple files per element).

        Parameters ::

            - pspots ('.+') : as defined above, will be a wildcard if not
                              specified.
            - elems (None) : set only these elements
            - notelems (None): do not set the elements
            - clear (True): clear previous settings
            - suffix (usp|UPF|recpot): PP file suffix
        """
    if clear and (not elems) and (not notelems):
        self.cell.species_pot.clear()
    if not os.path.isdir(self._castep_pp_path):
        if self._pedantic:
            warnings.warn('Cannot search directory: {} Folder does not exist'.format(self._castep_pp_path))
        return
    if pspot == '*':
        pspot = '.*'
    if suffix == '*':
        suffix = '.*'
    if pspot == '*':
        pspot = '.*'
    pattern = '^({elem}|{elem_upper}|{elem_lower})(_|-){pspot}\\.{suffix}$'
    for elem in set(self.atoms.get_chemical_symbols()):
        if elems is not None and elem not in elems:
            continue
        if notelems is not None and elem in notelems:
            continue
        p = pattern.format(elem=elem, elem_upper=elem.upper(), elem_lower=elem.lower(), pspot=pspot, suffix=suffix)
        pps = []
        for f in os.listdir(self._castep_pp_path):
            if re.match(p, f):
                pps.append(f)
        if not pps:
            if self._pedantic:
                warnings.warn('Pseudopotential for species {} not found!'.format(elem))
        elif not len(pps) == 1:
            raise RuntimeError('Pseudopotential for species {} not unique!\n'.format(elem) + 'Found the following files in {}\n'.format(self._castep_pp_path) + '\n'.join(['    {}'.format(pp) for pp in pps]) + '\nConsider a stricter search pattern in `find_pspots()`.')
        else:
            self.cell.species_pot = (elem, pps[0])