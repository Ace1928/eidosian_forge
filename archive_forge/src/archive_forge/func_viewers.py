import collections
import collections.abc
import operator
import warnings
@viewers.setter
def viewers(self, value):
    """Update viewers.

        Raise InvalidOperationException if version is greater than 1 or policy contains conditions.

        DEPRECATED:  use `policy.bindings` to modify bindings instead.
        """
    warnings.warn(_ASSIGNMENT_DEPRECATED_MSG.format('viewers', VIEWER_ROLE), DeprecationWarning)
    self[VIEWER_ROLE] = value