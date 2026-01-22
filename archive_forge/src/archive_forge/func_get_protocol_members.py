import abc
import collections
import collections.abc
import functools
import inspect
import operator
import sys
import types as _types
import typing
import warnings
def get_protocol_members(tp: type, /) -> typing.FrozenSet[str]:
    """Return the set of members defined in a Protocol.

        Example::

            >>> from typing_extensions import Protocol, get_protocol_members
            >>> class P(Protocol):
            ...     def a(self) -> str: ...
            ...     b: int
            >>> get_protocol_members(P)
            frozenset({'a', 'b'})

        Raise a TypeError for arguments that are not Protocols.
        """
    if not is_protocol(tp):
        raise TypeError(f'{tp!r} is not a Protocol')
    if hasattr(tp, '__protocol_attrs__'):
        return frozenset(tp.__protocol_attrs__)
    return frozenset(_get_protocol_attrs(tp))