from math import tanh, sqrt, exp
from operator import itemgetter
import numpy as np
from ase.db.core import now
from ase.ga import get_raw_score
def mass_extinction(self, ids):
    """Kills every candidate in the database with gaid in the
        supplied list of ids. Typically used on the main part of the current
        population if the diversity is to small.

        Parameters:

        ids: list
            list of ids of candidates to be killed.

        """
    for confid in ids:
        self.dc.kill_candidate(confid)
    self.pop = []