import collections
import collections.abc
import operator
import warnings
@editors.setter
def editors(self, value):
    """Update editors.

        Raise InvalidOperationException if version is greater than 1 or policy contains conditions.

        DEPRECATED:  use `policy.bindings` to modify bindings instead.
        """
    warnings.warn(_ASSIGNMENT_DEPRECATED_MSG.format('editors', EDITOR_ROLE), DeprecationWarning)
    self[EDITOR_ROLE] = value