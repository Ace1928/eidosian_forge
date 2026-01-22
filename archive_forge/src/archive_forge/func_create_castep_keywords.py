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
def create_castep_keywords(castep_command, filename='castep_keywords.json', force_write=True, path='.', fetch_only=None):
    """This function allows to fetch all available keywords from stdout
    of an installed castep binary. It furthermore collects the documentation
    to harness the power of (ipython) inspection and type for some basic
    type checking of input. All information is stored in a JSON file that is
    not distributed by default to avoid breaking the license of CASTEP.
    """
    suffixes = ['cell', 'param']
    filepath = os.path.join(path, filename)
    if os.path.exists(filepath) and (not force_write):
        warnings.warn('CASTEP Options Module file exists. You can overwrite it by calling python castep.py -f [CASTEP_COMMAND].')
        return False
    castep_version = get_castep_version(castep_command)
    help_all, _ = shell_stdouterr('%s -help all' % castep_command)
    try:
        if castep_version < 7.0:
            pattern = '((?<=^ )[A-Z_]{2,}|(?<=^)[A-Z_]{2,})'
        else:
            pattern = '((?<=^ )[A-Z_\\d]{2,}|(?<=^)[A-Z_\\d]{2,})'
        raw_options = re.findall(pattern, help_all, re.MULTILINE)
    except Exception:
        warnings.warn('Problem parsing: %s' % help_all)
        raise
    types = set()
    levels = set()
    processed_n = 0
    to_process = len(raw_options[:fetch_only])
    processed_options = {sf: {} for sf in suffixes}
    for o_i, option in enumerate(raw_options[:fetch_only]):
        doc, _ = shell_stdouterr('%s -help %s' % (castep_command, option))
        match = re.match('(?P<before_type>.*)Type: (?P<type>.+?)\\s+' + 'Level: (?P<level>[^ ]+)\\n\\s*\\n' + '(?P<doc>.*?)(\\n\\s*\\n|$)', doc, re.DOTALL)
        processed_n += 1
        if match is not None:
            match = match.groupdict()
            suffix = None
            if re.findall('PARAMETERS keywords:\\n\\n\\s?None found', doc):
                suffix = 'cell'
            if re.findall('CELL keywords:\\n\\n\\s?None found', doc):
                suffix = 'param'
            if suffix is None:
                warnings.warn('%s -> not assigned to either CELL or PARAMETERS keywords' % option)
            option = option.lower()
            mtyp = match.get('type', None)
            mlvl = match.get('level', None)
            mdoc = match.get('doc', None)
            if mtyp is None:
                warnings.warn('Found no type for %s' % option)
                continue
            if mlvl is None:
                warnings.warn('Found no level for %s' % option)
                continue
            if mdoc is None:
                warnings.warn('Found no doc string for %s' % option)
                continue
            types = types.union([mtyp])
            levels = levels.union([mlvl])
            processed_options[suffix][option] = {'keyword': option, 'option_type': mtyp, 'level': mlvl, 'docstring': mdoc}
            processed_n += 1
            frac = (o_i + 1.0) / to_process
            sys.stdout.write('\rProcessed: [{0}] {1:>3.0f}%'.format('#' * int(frac * 20) + ' ' * (20 - int(frac * 20)), 100 * frac))
            sys.stdout.flush()
        else:
            warnings.warn('create_castep_keywords: Could not process %s' % option)
    sys.stdout.write('\n')
    sys.stdout.flush()
    processed_options['types'] = list(types)
    processed_options['levels'] = list(levels)
    processed_options['castep_version'] = castep_version
    json.dump(processed_options, open(filepath, 'w'), indent=4)
    warnings.warn('CASTEP v%s, fetched %s keywords' % (castep_version, processed_n))
    return True