from math import tanh, sqrt, exp
from operator import itemgetter
import numpy as np
from ase.db.core import now
from ase.ga import get_raw_score
def get_nonrank(self, nrcand, key=None):
    """"Returns a list of fitness values."""
    nrc_list = []
    for nrc in nrcand:
        nrc_list.append(nrc.info['key_value_pairs'][key])
    return nrc_list