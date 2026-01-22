from typing import Any, Dict, Optional, Union, Iterable, List, Type, TYPE_CHECKING
from lazyops.utils.lazy import get_keydb_session
from .base import BaseStatefulBackend, SchemaType, logger
def decrbyfloat(self, key: str, amount: float=1.0, **kwargs) -> float:
    """
        [float] Decrements the value of the key by the given amount
        """
    if self.hset_enabled:
        return self.cache.hincrbyfloat(self.base_key, key, amount=amount * -1, **kwargs)
    return self.cache.incrbyfloat(self.get_key(key), amount=amount * -1, **kwargs)