from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import runtime
def step_into(break_on_async_call: typing.Optional[bool]=None) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Steps into the function call.

    :param break_on_async_call: **(EXPERIMENTAL)** *(Optional)* Debugger will pause on the execution of the first async task which was scheduled before next pause.
    """
    params: T_JSON_DICT = dict()
    if break_on_async_call is not None:
        params['breakOnAsyncCall'] = break_on_async_call
    cmd_dict: T_JSON_DICT = {'method': 'Debugger.stepInto', 'params': params}
    json = (yield cmd_dict)