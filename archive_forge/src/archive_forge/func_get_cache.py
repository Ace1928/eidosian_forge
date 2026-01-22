import asyncio
import functools
import hashlib
import os
from typing import Callable, Optional
import cloudpickle
from diskcache import Cache
@functools.lru_cache(1)
def get_cache():
    """Get the context object that contains previously-computed return values.

    The cache is used to avoid unnecessary computations and API calls, which can
    be long and expensive for large models.

    The cache directory defaults to `HOMEDIR/.cache/outlines`, but this choice
    can be overridden by the user by setting the value of the `OUTLINES_CACHE_DIR`
    environment variable.

    """
    from outlines._version import __version__ as outlines_version
    home_dir = os.path.expanduser('~')
    cache_dir = os.environ.get('OUTLINES_CACHE_DIR', f'{home_dir}/.cache/outlines')
    memory = Cache(cache_dir, eviction_policy='none', cull_limit=0)
    if outlines_version != memory.get('__version__'):
        memory.clear()
    memory['__version__'] = outlines_version
    return memory