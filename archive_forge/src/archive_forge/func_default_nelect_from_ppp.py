import os
import warnings
import shutil
from os.path import join, isfile, islink
from typing import List, Sequence, Tuple
import numpy as np
import ase
from ase.calculators.calculator import kpts2ndarray
from ase.calculators.vasp.setups import get_default_setups
def default_nelect_from_ppp(self):
    """ Get default number of electrons from ppp_list and symbol_count

        "Default" here means that the resulting cell would be neutral.
        """
    symbol_valences = []
    for filename in self.ppp_list:
        with open_potcar(filename=filename) as ppp_file:
            r = read_potcar_numbers_of_electrons(ppp_file)
            symbol_valences.extend(r)
    assert len(self.symbol_count) == len(symbol_valences)
    default_nelect = 0
    for (symbol1, count), (symbol2, valence) in zip(self.symbol_count, symbol_valences):
        assert symbol1 == symbol2
        default_nelect += count * valence
    return default_nelect