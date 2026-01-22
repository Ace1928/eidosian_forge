from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import debugger
from . import runtime
def take_precise_coverage() -> typing.Generator[T_JSON_DICT, T_JSON_DICT, typing.Tuple[typing.List[ScriptCoverage], float]]:
    """
    Collect coverage data for the current isolate, and resets execution counters. Precise code
    coverage needs to have started.

    :returns: A tuple with the following items:

        0. **result** - Coverage data for the current isolate.
        1. **timestamp** - Monotonically increasing time (in seconds) when the coverage update was taken in the backend.
    """
    cmd_dict: T_JSON_DICT = {'method': 'Profiler.takePreciseCoverage'}
    json = (yield cmd_dict)
    return ([ScriptCoverage.from_json(i) for i in json['result']], float(json['timestamp']))