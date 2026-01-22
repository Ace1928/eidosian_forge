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
def dryrun_ok(self, dryrun_flag='-dryrun'):
    """Starts a CASTEP run with the -dryrun flag [default]
        in a temporary and check wether all variables are initialized
        correctly. This is recommended for every bigger simulation.
        """
    from ase.io.castep import write_param
    temp_dir = tempfile.mkdtemp()
    self._fetch_pspots(temp_dir)
    seed = 'dryrun'
    self._write_cell(os.path.join(temp_dir, '%s.cell' % seed), self.atoms, castep_cell=self.cell)
    if not os.path.isfile(os.path.join(temp_dir, '%s.cell' % seed)):
        warnings.warn('%s.cell not written - aborting dryrun' % seed)
        return
    write_param(os.path.join(temp_dir, '%s.param' % seed), self.param)
    stdout, stderr = shell_stdouterr('%s %s %s' % (self._castep_command, seed, dryrun_flag), cwd=temp_dir)
    if stdout:
        print(stdout)
    if stderr:
        print(stderr)
    result_file = open(os.path.join(temp_dir, '%s.castep' % seed))
    txt = result_file.read()
    ok_string = '.*DRYRUN finished.*No problems found with input files.*'
    match = re.match(ok_string, txt, re.DOTALL)
    m = re.search('Number of kpoints used =\\s*([0-9]+)', txt)
    if m:
        self._kpoints = int(m.group(1))
    else:
        warnings.warn("Couldn't fetch number of kpoints from dryrun CASTEP file")
    err_file = os.path.join(temp_dir, '%s.0001.err' % seed)
    if match is None and os.path.exists(err_file):
        err_file = open(err_file)
        self._error = err_file.read()
        err_file.close()
    result_file.close()
    shutil.rmtree(temp_dir)
    return match is not None