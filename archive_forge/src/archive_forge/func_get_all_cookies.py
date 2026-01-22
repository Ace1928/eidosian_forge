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
def get_all_cookies() -> typing.Generator[T_JSON_DICT, T_JSON_DICT, typing.List[Cookie]]:
    """
    Returns all browser cookies. Depending on the backend support, will return detailed cookie
    information in the ``cookies`` field.
    Deprecated. Use Storage.getCookies instead.

    :returns: Array of cookie objects.
    """
    cmd_dict: T_JSON_DICT = {'method': 'Network.getAllCookies'}
    json = (yield cmd_dict)
    return [Cookie.from_json(i) for i in json['cookies']]