from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import target
def dispatch_periodic_sync_event(origin: str, registration_id: RegistrationID, tag: str) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    :param origin:
    :param registration_id:
    :param tag:
    """
    params: T_JSON_DICT = dict()
    params['origin'] = origin
    params['registrationId'] = registration_id.to_json()
    params['tag'] = tag
    cmd_dict: T_JSON_DICT = {'method': 'ServiceWorker.dispatchPeriodicSyncEvent', 'params': params}
    json = (yield cmd_dict)