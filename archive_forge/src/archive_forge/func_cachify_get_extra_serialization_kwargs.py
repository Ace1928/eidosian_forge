from __future__ import annotations
import abc
import contextlib
from kvdb.io import cachify as _cachify
from typing import Optional, Type, TypeVar, Union, Set, List, Any, Dict, Literal, TYPE_CHECKING
def cachify_get_extra_serialization_kwargs(self, func: str, **kwargs) -> Dict[str, Any]:
    """
        Gets the extra serialization kwargs
        """
    return {}