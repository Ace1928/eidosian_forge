import collections.abc
import typing
def is_generic_union(tp):
    """Returns true if `tp` is a parameterized typing.Union value."""
    return tp is not typing.Union and getattr(tp, '__origin__', None) is typing.Union