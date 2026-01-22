from __future__ import annotations
from .base import Always, Filter, FilterOrBool, Never
def to_filter(bool_or_filter: FilterOrBool) -> Filter:
    """
    Accept both booleans and Filters as input and
    turn it into a Filter.
    """
    if isinstance(bool_or_filter, bool):
        return _bool_to_filter[bool_or_filter]
    if isinstance(bool_or_filter, Filter):
        return bool_or_filter
    raise TypeError('Expecting a bool or a Filter instance. Got %r' % bool_or_filter)