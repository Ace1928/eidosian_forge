import collections.abc
import typing
def is_generic_mapping(tp):
    """Returns true if `tp` is a parameterized typing.Mapping value."""
    return tp not in (collections.abc.Mapping, typing.Mapping) and getattr(tp, '__origin__', None) in (collections.abc.Mapping, typing.Mapping)