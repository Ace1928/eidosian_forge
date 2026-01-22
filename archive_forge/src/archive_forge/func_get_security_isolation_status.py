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
def get_security_isolation_status(frame_id: typing.Optional[page.FrameId]=None) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, SecurityIsolationStatus]:
    """
    Returns information about the COEP/COOP isolation status.

    **EXPERIMENTAL**

    :param frame_id: *(Optional)* If no frameId is provided, the status of the target is provided.
    :returns: 
    """
    params: T_JSON_DICT = dict()
    if frame_id is not None:
        params['frameId'] = frame_id.to_json()
    cmd_dict: T_JSON_DICT = {'method': 'Network.getSecurityIsolationStatus', 'params': params}
    json = (yield cmd_dict)
    return SecurityIsolationStatus.from_json(json['status'])