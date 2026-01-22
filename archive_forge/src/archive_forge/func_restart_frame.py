from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import runtime
def restart_frame(call_frame_id: CallFrameId) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, typing.Tuple[typing.List[CallFrame], typing.Optional[runtime.StackTrace], typing.Optional[runtime.StackTraceId]]]:
    """
    Restarts particular call frame from the beginning.

    :param call_frame_id: Call frame identifier to evaluate on.
    :returns: A tuple with the following items:

        0. **callFrames** - New stack trace.
        1. **asyncStackTrace** - *(Optional)* Async stack trace, if any.
        2. **asyncStackTraceId** - *(Optional)* Async stack trace, if any.
    """
    params: T_JSON_DICT = dict()
    params['callFrameId'] = call_frame_id.to_json()
    cmd_dict: T_JSON_DICT = {'method': 'Debugger.restartFrame', 'params': params}
    json = (yield cmd_dict)
    return ([CallFrame.from_json(i) for i in json['callFrames']], runtime.StackTrace.from_json(json['asyncStackTrace']) if 'asyncStackTrace' in json else None, runtime.StackTraceId.from_json(json['asyncStackTraceId']) if 'asyncStackTraceId' in json else None)