import math
import textwrap
from abc import ABC, abstractmethod
import numpy as np
from scipy import spatial
from .._shared.utils import safe_as_int
from .._shared.compat import NP_COPY_IF_NEEDED
@property
def shear(self):
    if self.dimensionality != 2:
        raise NotImplementedError('The shear property is only implemented for 2D transforms.')
    beta = math.atan2(-self.params[0, 1], self.params[1, 1])
    return beta - self.rotation