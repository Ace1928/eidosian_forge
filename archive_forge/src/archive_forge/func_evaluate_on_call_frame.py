from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import runtime
def evaluate_on_call_frame(call_frame_id: CallFrameId, expression: str, object_group: typing.Optional[str]=None, include_command_line_api: typing.Optional[bool]=None, silent: typing.Optional[bool]=None, return_by_value: typing.Optional[bool]=None, generate_preview: typing.Optional[bool]=None, throw_on_side_effect: typing.Optional[bool]=None, timeout: typing.Optional[runtime.TimeDelta]=None) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, typing.Tuple[runtime.RemoteObject, typing.Optional[runtime.ExceptionDetails]]]:
    """
    Evaluates expression on a given call frame.

    :param call_frame_id: Call frame identifier to evaluate on.
    :param expression: Expression to evaluate.
    :param object_group: *(Optional)* String object group name to put result into (allows rapid releasing resulting object handles using ```releaseObjectGroup````).
    :param include_command_line_api: *(Optional)* Specifies whether command line API should be available to the evaluated expression, defaults to false.
    :param silent: *(Optional)* In silent mode exceptions thrown during evaluation are not reported and do not pause execution. Overrides ````setPauseOnException``` state.
    :param return_by_value: *(Optional)* Whether the result is expected to be a JSON object that should be sent by value.
    :param generate_preview: **(EXPERIMENTAL)** *(Optional)* Whether preview should be generated for the result.
    :param throw_on_side_effect: *(Optional)* Whether to throw an exception if side effect cannot be ruled out during evaluation.
    :param timeout: **(EXPERIMENTAL)** *(Optional)* Terminate execution after timing out (number of milliseconds).
    :returns: A tuple with the following items:

        0. **result** - Object wrapper for the evaluation result.
        1. **exceptionDetails** - *(Optional)* Exception details.
    """
    params: T_JSON_DICT = dict()
    params['callFrameId'] = call_frame_id.to_json()
    params['expression'] = expression
    if object_group is not None:
        params['objectGroup'] = object_group
    if include_command_line_api is not None:
        params['includeCommandLineAPI'] = include_command_line_api
    if silent is not None:
        params['silent'] = silent
    if return_by_value is not None:
        params['returnByValue'] = return_by_value
    if generate_preview is not None:
        params['generatePreview'] = generate_preview
    if throw_on_side_effect is not None:
        params['throwOnSideEffect'] = throw_on_side_effect
    if timeout is not None:
        params['timeout'] = timeout.to_json()
    cmd_dict: T_JSON_DICT = {'method': 'Debugger.evaluateOnCallFrame', 'params': params}
    json = (yield cmd_dict)
    return (runtime.RemoteObject.from_json(json['result']), runtime.ExceptionDetails.from_json(json['exceptionDetails']) if 'exceptionDetails' in json else None)