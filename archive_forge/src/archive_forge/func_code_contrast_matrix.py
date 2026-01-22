from __future__ import print_function
import sys
import six
import numpy as np
from patsy import PatsyError
from patsy.util import (repr_pretty_delegate, repr_pretty_impl,
def code_contrast_matrix(intercept, levels, contrast, default=None):
    if contrast is None:
        contrast = default
    if callable(contrast):
        contrast = contrast()
    if isinstance(contrast, ContrastMatrix):
        return contrast
    as_array = np.asarray(contrast)
    if safe_issubdtype(as_array.dtype, np.number):
        return ContrastMatrix(as_array, _name_levels('custom', range(as_array.shape[1])))
    if intercept:
        return contrast.code_with_intercept(levels)
    else:
        return contrast.code_without_intercept(levels)