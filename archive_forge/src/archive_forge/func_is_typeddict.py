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
def is_typeddict(tp):
    """Check if an annotation is a TypedDict class

        For example::
            class Film(TypedDict):
                title: str
                year: int

            is_typeddict(Film)  # => True
            is_typeddict(Union[list, str])  # => False
        """
    if hasattr(typing, 'TypedDict') and tp is typing.TypedDict:
        return False
    return isinstance(tp, _TYPEDDICT_TYPES)