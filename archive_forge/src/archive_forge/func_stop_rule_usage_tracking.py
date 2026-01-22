from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import page
def stop_rule_usage_tracking() -> typing.Generator[T_JSON_DICT, T_JSON_DICT, typing.List[RuleUsage]]:
    """
    Stop tracking rule usage and return the list of rules that were used since last call to
    ``takeCoverageDelta`` (or since start of coverage instrumentation).

    :returns: 
    """
    cmd_dict: T_JSON_DICT = {'method': 'CSS.stopRuleUsageTracking'}
    json = (yield cmd_dict)
    return [RuleUsage.from_json(i) for i in json['ruleUsage']]