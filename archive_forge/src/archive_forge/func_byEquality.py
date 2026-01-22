from ..helpers import map_values
from ._higherorder import (
from ._impl import Mismatch
@classmethod
def byEquality(cls, **kwargs):
    """Matches an object where the attributes equal the keyword values.

        Similar to the constructor, except that the matcher is assumed to be
        Equals.
        """
    from ._basic import Equals
    return cls.byMatcher(Equals, **kwargs)