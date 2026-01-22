import warnings
from warnings import warn
import breezy
def deprecated_passed(parameter_value):
    """Return True if parameter_value was used."""
    return parameter_value is not DEPRECATED_PARAMETER