from math import tanh, sqrt, exp
from operator import itemgetter
import numpy as np
from ase.db.core import now
from ase.ga import get_raw_score
def get_one_candidate(self):
    """Returns one candidates at random."""
    if len(self.pop) < 1:
        self.update()
    if len(self.pop) < 1:
        return None
    t = self.rng.randint(len(self.pop))
    c = self.pop[t]
    return c.copy()