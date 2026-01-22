from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import network
from . import page
def clear_device_metrics_override() -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Clears the overridden device metrics.
    """
    cmd_dict: T_JSON_DICT = {'method': 'Emulation.clearDeviceMetricsOverride'}
    json = (yield cmd_dict)