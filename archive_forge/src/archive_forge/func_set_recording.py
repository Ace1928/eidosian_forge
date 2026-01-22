from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import network
from . import service_worker
def set_recording(should_record: bool, service: ServiceName) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Set the recording state for the service.

    :param should_record:
    :param service:
    """
    params: T_JSON_DICT = dict()
    params['shouldRecord'] = should_record
    params['service'] = service.to_json()
    cmd_dict: T_JSON_DICT = {'method': 'BackgroundService.setRecording', 'params': params}
    json = (yield cmd_dict)