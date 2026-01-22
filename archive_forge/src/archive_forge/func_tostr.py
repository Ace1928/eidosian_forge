from warnings import warn
import numpy as np
from scipy._lib._util import VisibleDeprecationWarning
from ._sputils import (asmatrix, check_reshape_kwargs, check_shape,
from ._matrix import spmatrix
def tostr(row, col, data):
    triples = zip(list(zip(row, col)), data)
    return '\n'.join(['  {}\t{}'.format(*t) for t in triples])