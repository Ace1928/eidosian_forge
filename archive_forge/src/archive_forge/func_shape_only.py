from collections import namedtuple
from decimal import Decimal
import numpy as np
from . import backends, blas, helpers, parser, paths, sharing
def shape_only(shape):
    """Dummy ``numpy.ndarray`` which has a shape only - for generating
    contract expressions.
    """
    return Shaped(shape)