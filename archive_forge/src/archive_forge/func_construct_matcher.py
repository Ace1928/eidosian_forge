import types
from ._impl import (
def construct_matcher(*args, **kwargs):
    return _MatchesPredicateWithParams(predicate, message, name, *args, **kwargs)