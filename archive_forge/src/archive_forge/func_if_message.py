import types
from ._impl import (
@classmethod
def if_message(cls, annotation, matcher):
    """Annotate ``matcher`` only if ``annotation`` is non-empty."""
    if not annotation:
        return matcher
    return cls(annotation, matcher)