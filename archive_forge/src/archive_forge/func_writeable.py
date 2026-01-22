import sys
from . import errors as errors
from .identitymap import IdentityMap, NullIdentityMap
from .trace import mutter
def writeable(self):
    """Pass through transactions allow writes."""
    return True