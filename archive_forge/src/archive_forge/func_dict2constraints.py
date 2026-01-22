import os
import sys
import errno
import pickle
import warnings
import collections
import numpy as np
from ase.atoms import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.calculators.calculator import PropertyNotImplementedError
from ase.constraints import FixAtoms
from ase.parallel import world, barrier
def dict2constraints(d):
    """Convert dict unpickled from trajectory file to list of constraints."""
    version = d.get('version', 1)
    if version == 1:
        return d['constraints']
    elif version in (2, 3):
        try:
            constraints = pickle.loads(d['constraints_string'])
            for c in constraints:
                if isinstance(c, FixAtoms) and c.index.dtype == bool:
                    c.index = np.arange(len(c.index))[c.index]
            return constraints
        except (AttributeError, KeyError, EOFError, ImportError, TypeError):
            warnings.warn('Could not unpickle constraints!')
            return []
    else:
        return []