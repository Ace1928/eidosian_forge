from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import io
def record_clock_sync_marker(sync_id: str) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Record a clock sync marker in the trace.

    :param sync_id: The ID of this clock sync marker
    """
    params: T_JSON_DICT = dict()
    params['syncId'] = sync_id
    cmd_dict: T_JSON_DICT = {'method': 'Tracing.recordClockSyncMarker', 'params': params}
    json = (yield cmd_dict)