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
def key_val_str_to_dict(string, sep=None):
    """
    Parse an xyz properties string in a key=value and return a dict with
    various values parsed to native types.

    Accepts brackets or quotes to delimit values. Parses integers, floats
    booleans and arrays thereof. Arrays with 9 values whose name is listed
    in SPECIAL_3_3_KEYS are converted to 3x3 arrays with Fortran ordering.

    If sep is None, string will split on whitespace, otherwise will split
    key value pairs with the given separator.

    """
    delimiters = {"'": "'", '"': '"', '{': '}', '[': ']'}
    kv_pairs = [[[]]]
    cur_delimiter = None
    escaped = False
    for char in string.strip():
        if escaped:
            kv_pairs[-1][-1].append(char)
            escaped = False
        elif char == '\\':
            escaped = True
        elif cur_delimiter:
            if char == cur_delimiter:
                cur_delimiter = None
            else:
                kv_pairs[-1][-1].append(char)
        elif char in delimiters:
            cur_delimiter = delimiters[char]
        elif sep is None and char.isspace() or char == sep:
            if kv_pairs == [[[]]]:
                continue
            elif kv_pairs[-1][-1] == []:
                continue
            else:
                kv_pairs.append([[]])
        elif char == '=':
            if kv_pairs[-1] == [[]]:
                del kv_pairs[-1]
            kv_pairs[-1].append([])
        else:
            kv_pairs[-1][-1].append(char)
    kv_dict = {}
    for kv_pair in kv_pairs:
        if len(kv_pair) == 0:
            continue
        elif len(kv_pair) == 1:
            key, value = (''.join(kv_pair[0]), 'T')
        else:
            key, value = (''.join(kv_pair[0]), '='.join((''.join(x) for x in kv_pair[1:])))
        if key.lower() not in UNPROCESSED_KEYS:
            split_value = re.findall('[^\\s,]+', value)
            try:
                try:
                    numvalue = np.array(split_value, dtype=int)
                except (ValueError, OverflowError):
                    numvalue = np.array(split_value, dtype=float)
                if len(numvalue) == 1:
                    numvalue = numvalue[0]
                value = numvalue
            except (ValueError, OverflowError):
                pass
            if key in SPECIAL_3_3_KEYS:
                if not isinstance(value, np.ndarray) or value.shape != (9,):
                    raise ValueError('Got info item {}, expecting special 3x3 matrix, but value is not in the form of a 9-long numerical vector'.format(key))
                value = np.array(value).reshape((3, 3), order='F')
            if isinstance(value, str):
                str_to_bool = {'T': True, 'F': False}
                try:
                    boolvalue = [str_to_bool[vpart] for vpart in re.findall('[^\\s,]+', value)]
                    if len(boolvalue) == 1:
                        value = boolvalue[0]
                    else:
                        value = boolvalue
                except KeyError:
                    if value.startswith('_JSON '):
                        d = json.loads(value.replace('_JSON ', '', 1))
                        value = np.array(d)
                        if value.dtype.kind not in ['i', 'f', 'b']:
                            value = d
        kv_dict[key] = value
    return kv_dict