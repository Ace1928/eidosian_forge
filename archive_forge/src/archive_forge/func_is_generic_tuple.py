import collections.abc
import typing
def is_generic_tuple(tp):
    """Returns true if `tp` is a parameterized typing.Tuple value."""
    return tp not in (tuple, typing.Tuple) and getattr(tp, '__origin__', None) in (tuple, typing.Tuple)