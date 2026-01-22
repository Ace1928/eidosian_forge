import warnings
from breezy import symbol_versioning
from breezy.symbol_versioning import (deprecated_function, deprecated_in,
from breezy.tests import TestCase
@staticmethod
@deprecated_function(deprecated_in((0, 7, 0)))
def deprecated_static():
    """Deprecated static."""
    return 1