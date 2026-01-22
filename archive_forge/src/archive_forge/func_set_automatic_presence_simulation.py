from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
def set_automatic_presence_simulation(authenticator_id: AuthenticatorId, enabled: bool) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Sets whether tests of user presence will succeed immediately (if true) or fail to resolve (if false) for an authenticator.
    The default is true.

    :param authenticator_id:
    :param enabled:
    """
    params: T_JSON_DICT = dict()
    params['authenticatorId'] = authenticator_id.to_json()
    params['enabled'] = enabled
    cmd_dict: T_JSON_DICT = {'method': 'WebAuthn.setAutomaticPresenceSimulation', 'params': params}
    json = (yield cmd_dict)