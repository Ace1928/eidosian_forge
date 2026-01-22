from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import io
from . import network
from . import page
def fail_request(request_id: RequestId, error_reason: network.ErrorReason) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Causes the request to fail with specified reason.

    :param request_id: An id the client received in requestPaused event.
    :param error_reason: Causes the request to fail with the given reason.
    """
    params: T_JSON_DICT = dict()
    params['requestId'] = request_id.to_json()
    params['errorReason'] = error_reason.to_json()
    cmd_dict: T_JSON_DICT = {'method': 'Fetch.failRequest', 'params': params}
    json = (yield cmd_dict)