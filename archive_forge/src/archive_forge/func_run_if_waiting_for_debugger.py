from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
def run_if_waiting_for_debugger() -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Tells inspected instance to run if it was waiting for debugger to attach.
    """
    cmd_dict: T_JSON_DICT = {'method': 'Runtime.runIfWaitingForDebugger'}
    json = (yield cmd_dict)