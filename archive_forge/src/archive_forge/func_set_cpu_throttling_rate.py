from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import network
from . import page
def set_cpu_throttling_rate(rate: float) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Enables CPU throttling to emulate slow CPUs.

    :param rate: Throttling rate as a slowdown factor (1 is no throttle, 2 is 2x slowdown, etc).
    """
    params: T_JSON_DICT = dict()
    params['rate'] = rate
    cmd_dict: T_JSON_DICT = {'method': 'Emulation.setCPUThrottlingRate', 'params': params}
    json = (yield cmd_dict)