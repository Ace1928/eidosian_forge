from __future__ import (absolute_import, division, print_function)
import os
import sys
import numpy as np
from .util import banded_jacobian, sparse_jacobian_csc, sparse_jacobian_csr
def real_symarray(self, prefix, shape):
    return self.symarray(prefix, shape)