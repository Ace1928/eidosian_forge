from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
def start_tab_mirroring(sink_name: str) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Starts mirroring the tab to the sink.

    :param sink_name:
    """
    params: T_JSON_DICT = dict()
    params['sinkName'] = sink_name
    cmd_dict: T_JSON_DICT = {'method': 'Cast.startTabMirroring', 'params': params}
    json = (yield cmd_dict)