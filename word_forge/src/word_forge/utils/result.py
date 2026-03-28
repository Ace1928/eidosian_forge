from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Generic, Optional, TypeVar, Union

T = TypeVar("T")
E = TypeVar("E")
U = TypeVar("U")

@dataclass(frozen=True)
class Ok(Generic[T]):
    """Successful result container."""
    value: T
    
    @property
    def is_success(self) -> bool:
        return True
    
    @property
    def is_failure(self) -> bool:
        return False

    def unwrap(self) -> T:
        return self.value

    def unwrap_or(self, default: T) -> T:
        return self.value

    def map(self, func: Callable[[T], U]) -> Result[U, Any]:
        return Ok(func(self.value))

    def bind(self, func: Callable[[T], Result[U, E]]) -> Result[U, E]:
        return func(self.value)

@dataclass(frozen=True)
class Err(Generic[E]):
    """Failure result container."""
    error: E
    
    @property
    def is_success(self) -> bool:
        return False
    
    @property
    def is_failure(self) -> bool:
        return True

    def unwrap(self) -> Any:
        raise ValueError(f"Called unwrap on an Err value: {self.error}")

    def unwrap_or(self, default: T) -> T:
        return default

    def map(self, func: Callable[[Any], U]) -> Result[U, E]:
        return Err(self.error)

    def bind(self, func: Callable[[Any], Result[U, E]]) -> Result[U, E]:
        return Err(self.error)

Result = Union[Ok[T], Err[E]]

def success(value: T) -> Ok[T]:
    """Create a successful result."""
    return Ok(value)

def failure(error: E) -> Err[E]:
    """Create a failure result."""
    return Err(error)
