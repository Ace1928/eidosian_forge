import os
import warnings
import shutil
from os.path import join, isfile, islink
from typing import List, Sequence, Tuple
import numpy as np
import ase
from ase.calculators.calculator import kpts2ndarray
from ase.calculators.vasp.setups import get_default_setups
def set_xc_params(self, xc):
    """Set parameters corresponding to XC functional"""
    xc = xc.lower()
    if xc is None:
        pass
    elif xc not in self.xc_defaults:
        xc_allowed = ', '.join(self.xc_defaults.keys())
        raise ValueError('{0} is not supported for xc! Supported xc valuesare: {1}'.format(xc, xc_allowed))
    else:
        if 'pp' not in self.xc_defaults[xc]:
            self.set(pp='PBE')
        self.set(**self.xc_defaults[xc])