from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import runtime
def set_script_source(script_id: runtime.ScriptId, script_source: str, dry_run: typing.Optional[bool]=None) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, typing.Tuple[typing.Optional[typing.List[CallFrame]], typing.Optional[bool], typing.Optional[runtime.StackTrace], typing.Optional[runtime.StackTraceId], typing.Optional[runtime.ExceptionDetails]]]:
    """
    Edits JavaScript source live.

    :param script_id: Id of the script to edit.
    :param script_source: New content of the script.
    :param dry_run: *(Optional)* If true the change will not actually be applied. Dry run may be used to get result description without actually modifying the code.
    :returns: A tuple with the following items:

        0. **callFrames** - *(Optional)* New stack trace in case editing has happened while VM was stopped.
        1. **stackChanged** - *(Optional)* Whether current call stack  was modified after applying the changes.
        2. **asyncStackTrace** - *(Optional)* Async stack trace, if any.
        3. **asyncStackTraceId** - *(Optional)* Async stack trace, if any.
        4. **exceptionDetails** - *(Optional)* Exception details if any.
    """
    params: T_JSON_DICT = dict()
    params['scriptId'] = script_id.to_json()
    params['scriptSource'] = script_source
    if dry_run is not None:
        params['dryRun'] = dry_run
    cmd_dict: T_JSON_DICT = {'method': 'Debugger.setScriptSource', 'params': params}
    json = (yield cmd_dict)
    return ([CallFrame.from_json(i) for i in json['callFrames']] if 'callFrames' in json else None, bool(json['stackChanged']) if 'stackChanged' in json else None, runtime.StackTrace.from_json(json['asyncStackTrace']) if 'asyncStackTrace' in json else None, runtime.StackTraceId.from_json(json['asyncStackTraceId']) if 'asyncStackTraceId' in json else None, runtime.ExceptionDetails.from_json(json['exceptionDetails']) if 'exceptionDetails' in json else None)