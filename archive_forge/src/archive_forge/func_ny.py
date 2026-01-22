from __future__ import absolute_import, division, print_function
from collections import OrderedDict
from itertools import repeat, chain
import warnings
import numpy as np
from .util import import_
from .core import ODESys, RecoverableError
from .util import (
@property
def ny(self):
    """ Number of dependent variables in the system. """
    return len(self.exprs)