import sys
from . import errors as errors
from .identitymap import IdentityMap, NullIdentityMap
from .trace import mutter
def register_dirty(self, an_object):
    """Register an_object as being dirty.

        Dirty objects get informed
        when the transaction finishes.
        """
    self._dirty_objects.add(an_object)