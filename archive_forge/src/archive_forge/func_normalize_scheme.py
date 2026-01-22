import re
from . import compat
from . import misc
def normalize_scheme(scheme):
    """Normalize the scheme component."""
    return scheme.lower()