import collections
from abc import ABC, abstractmethod
import numpy as np
from scipy._lib._util import MapWrapper
def proc_minimisers(self):
    """Check for minimisers."""
    for v in self:
        v.minimiser()
        v.maximiser()