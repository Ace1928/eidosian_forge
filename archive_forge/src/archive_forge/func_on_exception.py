from __future__ import annotations
from typing import Any, Callable, NamedTuple
from pydantic_core import CoreConfig, CoreSchema, ValidationError
from typing_extensions import Literal, Protocol, TypeAlias
def on_exception(self, exception: Exception) -> None:
    """Callback to be notified of validation exceptions.

        Args:
            exception: The exception raised during validation.
        """
    return