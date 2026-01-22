from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import runtime
def pause_on_async_call(parent_stack_trace_id: runtime.StackTraceId) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """


    **EXPERIMENTAL**

    :param parent_stack_trace_id: Debugger will pause when async call with given stack trace is started.
    """
    params: T_JSON_DICT = dict()
    params['parentStackTraceId'] = parent_stack_trace_id.to_json()
    cmd_dict: T_JSON_DICT = {'method': 'Debugger.pauseOnAsyncCall', 'params': params}
    json = (yield cmd_dict)