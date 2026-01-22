from __future__ import annotations
import contextlib
import datetime
from functools import partial
from functools import wraps
import json
import logging
from numbers import Number
import threading
import time
from typing import Any
from typing import Callable
from typing import cast
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import Union
from decorator import decorate
from . import exception
from .api import BackendArguments
from .api import BackendFormatted
from .api import CachedValue
from .api import CacheMutex
from .api import CacheReturnType
from .api import CantDeserializeException
from .api import KeyType
from .api import MetaDataType
from .api import NO_VALUE
from .api import SerializedReturnType
from .api import Serializer
from .api import ValuePayload
from .backends import _backend_loader
from .backends import register_backend  # noqa
from .proxy import ProxyBackend
from .util import function_key_generator
from .util import function_multi_key_generator
from .util import repr_obj
from .. import Lock
from .. import NeedRegenerationException
from ..util import coerce_string_conf
from ..util import memoized_property
from ..util import NameRegistry
from ..util import PluginLoader
from ..util.typing import Self
def get_or_create_for_user_func(key_generator: Callable[..., Sequence[KeyType]], user_func: Callable[..., Sequence[ValuePayload]], *arg: Any, **kw: Any) -> Union[Sequence[ValuePayload], Mapping[KeyType, ValuePayload]]:
    cache_keys = arg
    keys = key_generator(*arg, **kw)
    key_lookup = dict(zip(keys, cache_keys))

    @wraps(user_func)
    def creator(*keys_to_create):
        return user_func(*[key_lookup[k] for k in keys_to_create])
    timeout: Optional[float] = cast(ExpirationTimeCallable, expiration_time)() if expiration_time_is_callable else cast(Optional[float], expiration_time)
    result: Union[Sequence[ValuePayload], Mapping[KeyType, ValuePayload]]
    if asdict:

        def dict_create(*keys):
            d_values = creator(*keys)
            return [d_values.get(key_lookup[k], NO_VALUE) for k in keys]

        def wrap_cache_fn(value):
            if value is NO_VALUE:
                return False
            elif not should_cache_fn:
                return True
            else:
                return should_cache_fn(value)
        result = self.get_or_create_multi(keys, dict_create, timeout, wrap_cache_fn)
        result = dict(((k, v) for k, v in zip(cache_keys, result) if v is not NO_VALUE))
    else:
        result = self.get_or_create_multi(keys, creator, timeout, should_cache_fn)
    return result