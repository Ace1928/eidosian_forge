import os
import warnings
import shutil
from os.path import join, isfile, islink
from typing import List, Sequence, Tuple
import numpy as np
import ase
from ase.calculators.calculator import kpts2ndarray
from ase.calculators.vasp.setups import get_default_setups
def check_xc(self):
    """Make sure the calculator has functional & pseudopotentials set up

        If no XC combination, GGA functional or POTCAR type is specified,
        default to PW91. Otherwise, try to guess the desired pseudopotentials.
        """
    p = self.input_params
    if 'pp' not in p or p['pp'] is None:
        if self.string_params['gga'] is None:
            p.update({'pp': 'lda'})
        elif self.string_params['gga'] == '91':
            p.update({'pp': 'pw91'})
        elif self.string_params['gga'] == 'PE':
            p.update({'pp': 'pbe'})
        else:
            raise NotImplementedError("Unable to guess the desired set of pseudopotential(POTCAR) files. Please do one of the following: \n1. Use the 'xc' parameter to define your XC functional.These 'recipes' determine the pseudopotential file as well as setting the INCAR parameters.\n2. Use the 'gga' settings None (default), 'PE' or '91'; these correspond to LDA, PBE and PW91 respectively.\n3. Set the POTCAR explicitly with the 'pp' flag. The value should be the name of a folder on the VASP_PP_PATH, and the aliases 'LDA', 'PBE' and 'PW91' are alsoaccepted.\n")
    if p['xc'] is not None and p['xc'].lower() == 'lda' and (p['pp'].lower() != 'lda'):
        warnings.warn('XC is set to LDA, but PP is set to {0}. \nThis calculation is using the {0} POTCAR set. \n Please check that this is really what you intended!\n'.format(p['pp'].upper()))