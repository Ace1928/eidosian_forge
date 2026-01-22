import collections
from abc import ABC, abstractmethod
import numpy as np
from scipy._lib._util import MapWrapper
def proc_fpool_g(self):
    """Process all field functions with constraints supplied."""
    for v in self.fpool:
        if v.feasible:
            self.compute_sfield(v)
    self.fpool = set()