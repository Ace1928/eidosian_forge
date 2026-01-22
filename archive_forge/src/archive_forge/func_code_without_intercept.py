from __future__ import print_function
import sys
import six
import numpy as np
from patsy import PatsyError
from patsy.util import (repr_pretty_delegate, repr_pretty_impl,
def code_without_intercept(self, levels):
    contrast = self._diff_contrast(levels)
    return ContrastMatrix(contrast, _name_levels('D.', levels[:-1]))