import re
import warnings
import weakref
from collections import OrderedDict
from .. import functions as fn
from ..Qt import QtCore
from .ParameterItem import ParameterItem
def defaultValue(self):
    """Return the default value for this parameter. Raises ValueError if no default."""
    if 'default' not in self.opts:
        warnings.warn('Parameter has no default value. This will be a ValueError after January 2025.', DeprecationWarning, stacklevel=2)
    return self.opts.get('default')