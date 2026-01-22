import collections
import collections.abc
import operator
import warnings
@owners.setter
def owners(self, value):
    """Update owners.

        Raise InvalidOperationException if version is greater than 1 or policy contains conditions.

        DEPRECATED:  use `policy.bindings` to access bindings instead.
        """
    warnings.warn(_ASSIGNMENT_DEPRECATED_MSG.format('owners', OWNER_ROLE), DeprecationWarning)
    self[OWNER_ROLE] = value