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
def import_castep_keywords(castep_command='', filename='castep_keywords.json', path='.'):
    searchpaths = [path, os.path.expanduser('~/.ase'), os.path.join(ase.__path__[0], 'calculators')]
    try:
        kwfile = sum([glob.glob(os.path.join(sp, filename)) for sp in searchpaths], [])[0]
    except IndexError:
        warnings.warn('Generating CASTEP keywords JSON file... hang on.\n    The CASTEP keywords JSON file contains abstractions for CASTEP input\n    parameters (for both .cell and .param input files), including some\n    format checks and descriptions. The latter are extracted from the\n    internal online help facility of a CASTEP binary, thus allowing to\n    easily keep the calculator synchronized with (different versions of)\n    the CASTEP code. Consequently, avoiding licensing issues (CASTEP is\n    distributed commercially by accelrys), we consider it wise not to\n    provide the file in the first place.')
        create_castep_keywords(get_castep_command(castep_command), filename=filename, path=path)
        warnings.warn('Stored %s in %s.  Copy it to your ASE installation under ase/calculators for system-wide installation. Using a *nix OS this can be a simple as mv %s %s' % (filename, os.path.abspath(path), os.path.join(os.path.abspath(path), filename), os.path.join(os.path.dirname(ase.__file__), 'calculators')))
        kwfile = os.path.join(path, filename)
    kwdata = json.load(open(kwfile))
    param_dict = make_param_dict(kwdata['param'])
    cell_dict = make_cell_dict(kwdata['cell'])
    castep_keywords = CastepKeywords(param_dict, cell_dict, kwdata['types'], kwdata['levels'], kwdata['castep_version'])
    return castep_keywords