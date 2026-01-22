import logging
import types
import weakref
from dataclasses import dataclass
from . import config
def is_recompilation(cache_size: CacheSizeRelevantForFrame) -> bool:
    """
    If the frame (earlier parsed by compute_cache_size) has more than 1 cache
    entry with same ID_MATCH'd objects, then its a recompilation.
    """
    return cache_size.will_compilation_exceed(1)