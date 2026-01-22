from __future__ import annotations
from typing import TYPE_CHECKING, Any, Callable, Union
def is_dimension(value: object) -> TypeGuard[AnyDimension]:
    """
    Test whether the given value could be a valid dimension.
    (For usage in an assertion. It's not guaranteed in case of a callable.)
    """
    if value is None:
        return True
    if callable(value):
        return True
    if isinstance(value, (int, Dimension)):
        return True
    return False