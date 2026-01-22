from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Union
from ..containers import PrimitiveResult
from .base_result import _BasePrimitiveResult
@abstractmethod
def in_final_state(self) -> bool:
    """Return whether the job is in a final job state such as ``DONE`` or ``ERROR``."""
    raise NotImplementedError('Subclass of BasePrimitiveJob must implement `is_final_state` method.')