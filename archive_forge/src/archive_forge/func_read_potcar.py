import os
import warnings
import shutil
from os.path import join, isfile, islink
from typing import List, Sequence, Tuple
import numpy as np
import ase
from ase.calculators.calculator import kpts2ndarray
from ase.calculators.vasp.setups import get_default_setups
def read_potcar(self, filename):
    """ Read the pseudopotential XC functional from POTCAR file.
        """
    xc_flag = None
    with open(filename, 'r') as fd:
        for line in fd:
            key = line.split()[0].upper()
            if key == 'LEXCH':
                xc_flag = line.split()[-1].upper()
                break
    if xc_flag is None:
        raise ValueError('LEXCH flag not found in POTCAR file.')
    xc_dict = {'PE': 'PBE', '91': 'PW91', 'CA': 'LDA'}
    if xc_flag not in xc_dict.keys():
        raise ValueError('Unknown xc-functional flag found in POTCAR, LEXCH=%s' % xc_flag)
    self.input_params['pp'] = xc_dict[xc_flag]