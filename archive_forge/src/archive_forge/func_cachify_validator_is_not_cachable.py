from __future__ import annotations
import abc
import contextlib
from kvdb.io import cachify as _cachify
from typing import Optional, Type, TypeVar, Union, Set, List, Any, Dict, Literal, TYPE_CHECKING
def cachify_validator_is_not_cachable(self, *args, cachable: Optional[bool]=True, **kwargs) -> bool:
    """
        Checks if the function is not cachable
        """
    from kvdb.io.cachify.helpers import is_not_cachable
    return is_not_cachable(*args, cachable=cachable, **kwargs)