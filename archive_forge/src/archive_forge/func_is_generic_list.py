import collections.abc
import typing
def is_generic_list(tp):
    """Returns true if `tp` is a parameterized typing.List value."""
    return tp not in (list, typing.List) and getattr(tp, '__origin__', None) in (list, typing.List)