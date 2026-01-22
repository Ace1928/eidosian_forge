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
def known_types_to_str(val):
    if isinstance(val, bool) or isinstance(val, np.bool_):
        return 'T' if val else 'F'
    elif isinstance(val, numbers.Real):
        return '{}'.format(val)
    elif isinstance(val, Spacegroup):
        return val.symbol
    else:
        return val