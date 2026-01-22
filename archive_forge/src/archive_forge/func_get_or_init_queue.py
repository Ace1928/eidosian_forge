from typing import Dict, TypeVar, Optional, Union, Any, TYPE_CHECKING
from .registry import (
from lazyops.types.lazydict import LazyDict, RT
def get_or_init_queue(cls, queue_name: str, kind: Optional[str]=None) -> 'TaskQueue':
    """
        Returns the queue object
        """
    if queue_name not in cls._queues:
        if kind is None:
            for k, v in cls.queue_schemas.items():
                if queue_name in v:
                    kind = k
                    break
        try:
            q = cls.get_task_queue(name=queue_name, kind=kind)
        except Exception as e:
            q = False
        cls._queues[queue_name] = q
    return cls._queues[queue_name]