from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import browser
from . import page
def send_message_to_target(message: str, session_id: typing.Optional[SessionID]=None, target_id: typing.Optional[TargetID]=None) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Sends protocol message over session with given id.
    Consider using flat mode instead; see commands attachToTarget, setAutoAttach,
    and crbug.com/991325.

    :param message:
    :param session_id: *(Optional)* Identifier of the session.
    :param target_id: *(Optional)* Deprecated.
    """
    params: T_JSON_DICT = dict()
    params['message'] = message
    if session_id is not None:
        params['sessionId'] = session_id.to_json()
    if target_id is not None:
        params['targetId'] = target_id.to_json()
    cmd_dict: T_JSON_DICT = {'method': 'Target.sendMessageToTarget', 'params': params}
    json = (yield cmd_dict)