from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import debugger
from . import emulation
from . import io
from . import page
from . import runtime
from . import security
def set_cache_disabled(cache_disabled: bool) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Toggles ignoring cache for each request. If ``true``, cache will not be used.

    :param cache_disabled: Cache disabled state.
    """
    params: T_JSON_DICT = dict()
    params['cacheDisabled'] = cache_disabled
    cmd_dict: T_JSON_DICT = {'method': 'Network.setCacheDisabled', 'params': params}
    json = (yield cmd_dict)