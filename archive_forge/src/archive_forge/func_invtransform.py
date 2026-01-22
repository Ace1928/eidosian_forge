import math
import numpy as np
from scipy import linalg, stats, special
from .linalg_decomp_1 import SvdArray
def invtransform(self, y):
    return np.dot(self.tmatinv, y - self.const)