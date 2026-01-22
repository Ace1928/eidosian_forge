import os
import numpy as np
from ase.units import Bohr, Ha, Ry, fs, m, s
from ase.calculators.calculator import kpts2sizeandoffsets
from ase.calculators.openmx.reader import (read_electron_valency, get_file_name, get_standard_key)
from ase.calculators.openmx import parameters as param
def get_xc(xc):
    """
    Change the name of xc appropriate to OpenMX format
    """
    xc = xc.upper()
    assert xc.upper() in param.OpenMXParameters().allowed_xc
    if xc in ['PBE', 'GGA', 'GGA-PBE']:
        return 'GGA-PBE'
    elif xc in ['LDA']:
        return 'LDA'
    elif xc in ['CA', 'PW']:
        return 'LSDA-' + xc
    elif xc in ['LSDA', 'LSDA-CA']:
        return 'LSDA-CA'
    elif xc in ['LSDA-PW']:
        return 'LSDA-PW'
    else:
        return 'LDA'