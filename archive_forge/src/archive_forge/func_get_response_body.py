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
def get_response_body(request_id: RequestId) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, typing.Tuple[str, bool]]:
    """
    Returns content served for the given request.

    :param request_id: Identifier of the network request to get content for.
    :returns: A tuple with the following items:

        0. **body** - Response body.
        1. **base64Encoded** - True, if content was sent as base64.
    """
    params: T_JSON_DICT = dict()
    params['requestId'] = request_id.to_json()
    cmd_dict: T_JSON_DICT = {'method': 'Network.getResponseBody', 'params': params}
    json = (yield cmd_dict)
    return (str(json['body']), bool(json['base64Encoded']))