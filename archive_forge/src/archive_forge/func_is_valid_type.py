from __future__ import annotations
from pydantic import BaseModel
from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Any, Dict, Optional, Iterable, Type, TYPE_CHECKING
def is_valid_type(self, data: Any) -> bool:
    """
        Returns True if the Data is a Valid Type
        """
    return self.schema_type is None or isinstance(data, self.schema_type)