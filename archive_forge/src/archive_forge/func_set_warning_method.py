import warnings
from warnings import warn
import breezy
def set_warning_method(method):
    """Set the warning method to be used by this module.

    It should take a message and a warning category as warnings.warn does.
    """
    global warn
    warn = method