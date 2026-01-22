from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import browser
from . import network
from . import page
def run_bounce_tracking_mitigations() -> typing.Generator[T_JSON_DICT, T_JSON_DICT, typing.List[str]]:
    """
    Deletes state for sites identified as potential bounce trackers, immediately.

    **EXPERIMENTAL**

    :returns: 
    """
    cmd_dict: T_JSON_DICT = {'method': 'Storage.runBounceTrackingMitigations'}
    json = (yield cmd_dict)
    return [str(i) for i in json['deletedSites']]