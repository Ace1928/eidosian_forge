import io
import os
import numpy as np
import scipy.sparse
from scipy.io import _mmio
def read1(self, size=-1):
    return self._encoding_call('read1', size)