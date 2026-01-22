from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
def set_sink_to_use(sink_name: str) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Sets a sink to be used when the web page requests the browser to choose a
    sink via Presentation API, Remote Playback API, or Cast SDK.

    :param sink_name:
    """
    params: T_JSON_DICT = dict()
    params['sinkName'] = sink_name
    cmd_dict: T_JSON_DICT = {'method': 'Cast.setSinkToUse', 'params': params}
    json = (yield cmd_dict)