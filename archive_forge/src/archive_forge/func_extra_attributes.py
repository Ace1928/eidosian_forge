from __future__ import annotations
from collections.abc import Callable, Mapping
from typing import Any, TypeVar, final, overload
from ._exceptions import TypedAttributeLookupError
@property
def extra_attributes(self) -> Mapping[T_Attr, Callable[[], T_Attr]]:
    """
        A mapping of the extra attributes to callables that return the corresponding
        values.

        If the provider wraps another provider, the attributes from that wrapper should
        also be included in the returned mapping (but the wrapper may override the
        callables from the wrapped instance).

        """
    return {}