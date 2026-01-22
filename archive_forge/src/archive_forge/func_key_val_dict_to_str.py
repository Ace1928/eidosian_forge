from itertools import islice
import re
import warnings
from io import StringIO, UnsupportedOperation
import json
import numpy as np
import numbers
from ase.atoms import Atoms
from ase.calculators.calculator import all_properties, Calculator
from ase.calculators.singlepoint import SinglePointCalculator
from ase.spacegroup.spacegroup import Spacegroup
from ase.parallel import paropen
from ase.constraints import FixAtoms, FixCartesian
from ase.io.formats import index2range
from ase.utils import reader
def key_val_dict_to_str(dct, sep=' '):
    """
    Convert atoms.info dictionary to extended XYZ string representation
    """

    def array_to_string(key, val):
        if key in SPECIAL_3_3_KEYS:
            val = val.reshape(val.size, order='F')
        if val.dtype.kind in ['i', 'f', 'b']:
            if len(val.shape) == 0:
                val = str(known_types_to_str(val))
            elif len(val.shape) == 1:
                val = ' '.join((str(known_types_to_str(v)) for v in val))
        return val

    def known_types_to_str(val):
        if isinstance(val, bool) or isinstance(val, np.bool_):
            return 'T' if val else 'F'
        elif isinstance(val, numbers.Real):
            return '{}'.format(val)
        elif isinstance(val, Spacegroup):
            return val.symbol
        else:
            return val
    if len(dct) == 0:
        return ''
    string = ''
    for key in dct:
        val = dct[key]
        if isinstance(val, np.ndarray):
            val = array_to_string(key, val)
        else:
            val = known_types_to_str(val)
        if val is not None and (not isinstance(val, str)):
            if isinstance(val, np.ndarray):
                val = val.tolist()
            try:
                val = '_JSON ' + json.dumps(val)
            except TypeError:
                warnings.warn('Skipping unhashable information {0}'.format(key))
                continue
        key = escape(key)
        eq = '='
        if val is None:
            val = ''
            eq = ''
        val = escape(val)
        string += '%s%s%s%s' % (key, eq, val, sep)
    return string.strip()