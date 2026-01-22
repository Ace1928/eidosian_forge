from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import runtime
def set_variable_value(scope_number: int, variable_name: str, new_value: runtime.CallArgument, call_frame_id: CallFrameId) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Changes value of variable in a callframe. Object-based scopes are not supported and must be
    mutated manually.

    :param scope_number: 0-based number of scope as was listed in scope chain. Only 'local', 'closure' and 'catch' scope types are allowed. Other scopes could be manipulated manually.
    :param variable_name: Variable name.
    :param new_value: New variable value.
    :param call_frame_id: Id of callframe that holds variable.
    """
    params: T_JSON_DICT = dict()
    params['scopeNumber'] = scope_number
    params['variableName'] = variable_name
    params['newValue'] = new_value.to_json()
    params['callFrameId'] = call_frame_id.to_json()
    cmd_dict: T_JSON_DICT = {'method': 'Debugger.setVariableValue', 'params': params}
    json = (yield cmd_dict)