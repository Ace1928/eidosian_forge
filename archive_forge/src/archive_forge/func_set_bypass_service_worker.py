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
def set_bypass_service_worker(bypass: bool) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Toggles ignoring of service worker for each request.

    :param bypass: Bypass service worker and load from network.
    """
    params: T_JSON_DICT = dict()
    params['bypass'] = bypass
    cmd_dict: T_JSON_DICT = {'method': 'Network.setBypassServiceWorker', 'params': params}
    json = (yield cmd_dict)