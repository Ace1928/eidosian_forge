from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import page
def start_rule_usage_tracking() -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Enables the selector recording.
    """
    cmd_dict: T_JSON_DICT = {'method': 'CSS.startRuleUsageTracking'}
    json = (yield cmd_dict)