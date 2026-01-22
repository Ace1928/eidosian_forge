from copy import deepcopy
import re
import numpy as np
from ... import config
from ...interfaces.base import DynamicTraitedSpec
from ...utils.filemanip import loadpkl, savepkl
@property
def fullname(self):
    """Build the full name down the hierarchy."""
    if self._hierarchy:
        return '%s.%s' % (self._hierarchy, self.name)
    return self.name