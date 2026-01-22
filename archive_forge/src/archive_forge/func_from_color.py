import sys
import warnings
from ..overrides import override, strip_boolean_result
from ..module import get_introspection_module
from gi import PyGIDeprecationWarning, require_version
@classmethod
def from_color(cls, color):
    """Returns a new RGBA instance given a Color instance."""
    return cls(color.red_float, color.green_float, color.blue_float)