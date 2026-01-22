from typing import Dict, TypeVar, Optional, Union, Any, TYPE_CHECKING
from .registry import (
from lazyops.types.lazydict import LazyDict, RT
@property
def queue_schemas(cls) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
        Returns the queue schemas
        """
    if cls._queue_schemas is None:
        cls._queue_schemas = cls.get_queue_schemas()
    return cls._queue_schemas