import sys
import warnings
from ..overrides import override, strip_boolean_result
from ..module import get_introspection_module
from gi import PyGIDeprecationWarning, require_version
def to_floats(self):
    """Return (red_float, green_float, blue_float) triple."""
    return (self.red_float, self.green_float, self.blue_float)