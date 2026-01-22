from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import runtime
def get_wasm_bytecode(script_id: runtime.ScriptId) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, str]:
    """
    This command is deprecated. Use getScriptSource instead.

    :param script_id: Id of the Wasm script to get source for.
    :returns: Script source.
    """
    params: T_JSON_DICT = dict()
    params['scriptId'] = script_id.to_json()
    cmd_dict: T_JSON_DICT = {'method': 'Debugger.getWasmBytecode', 'params': params}
    json = (yield cmd_dict)
    return str(json['bytecode'])