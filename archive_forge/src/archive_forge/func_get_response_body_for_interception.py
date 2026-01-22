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
def get_response_body_for_interception(interception_id: InterceptionId) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, typing.Tuple[str, bool]]:
    """
    Returns content served for the given currently intercepted request.

    **EXPERIMENTAL**

    :param interception_id: Identifier for the intercepted request to get body for.
    :returns: A tuple with the following items:

        0. **body** - Response body.
        1. **base64Encoded** - True, if content was sent as base64.
    """
    params: T_JSON_DICT = dict()
    params['interceptionId'] = interception_id.to_json()
    cmd_dict: T_JSON_DICT = {'method': 'Network.getResponseBodyForInterception', 'params': params}
    json = (yield cmd_dict)
    return (str(json['body']), bool(json['base64Encoded']))