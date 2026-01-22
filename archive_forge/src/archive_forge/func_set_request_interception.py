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
def set_request_interception(patterns: typing.List[RequestPattern]) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Sets the requests to intercept that match the provided patterns and optionally resource types.
    Deprecated, please use Fetch.enable instead.

    **EXPERIMENTAL**

    :param patterns: Requests matching any of these patterns will be forwarded and wait for the corresponding continueInterceptedRequest call.
    """
    params: T_JSON_DICT = dict()
    params['patterns'] = [i.to_json() for i in patterns]
    cmd_dict: T_JSON_DICT = {'method': 'Network.setRequestInterception', 'params': params}
    json = (yield cmd_dict)