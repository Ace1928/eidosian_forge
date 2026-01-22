from __future__ import print_function
import warnings
import numbers
import six
import numpy as np
from patsy import PatsyError
from patsy.util import atleast_2d_column_default
from patsy.compat import OrderedDict
from patsy.util import (repr_pretty_delegate, repr_pretty_impl,
from patsy.constraint import linear_constraint
from patsy.contrasts import ContrastMatrix
from patsy.desc import ModelDesc, Term
@property
def design_info(self):
    """.. deprecated:: 0.4.0"""
    warnings.warn(DeprecationWarning("Starting in patsy v0.4.0, the DesignMatrixBuilder class has been merged into the DesignInfo class. So there's no need to use builder.design_info to access the DesignInfo; 'builder' already *is* a DesignInfo."), stacklevel=2)
    return self