from typing import Dict, TypeVar, Optional, Union, Any, TYPE_CHECKING
from .registry import (
from lazyops.types.lazydict import LazyDict, RT
def get_queue_schemas(cls) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
        Returns the queue schemas

        - Should be overwritten by the subclass
        """
    raise NotImplementedError