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
def take_response_body_for_interception_as_stream(interception_id: InterceptionId) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, io.StreamHandle]:
    """
    Returns a handle to the stream representing the response body. Note that after this command,
    the intercepted request can't be continued as is -- you either need to cancel it or to provide
    the response body. The stream only supports sequential read, IO.read will fail if the position
    is specified.

    **EXPERIMENTAL**

    :param interception_id:
    :returns: 
    """
    params: T_JSON_DICT = dict()
    params['interceptionId'] = interception_id.to_json()
    cmd_dict: T_JSON_DICT = {'method': 'Network.takeResponseBodyForInterceptionAsStream', 'params': params}
    json = (yield cmd_dict)
    return io.StreamHandle.from_json(json['stream'])