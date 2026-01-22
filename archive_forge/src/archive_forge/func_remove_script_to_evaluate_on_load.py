from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import debugger
from . import dom
from . import emulation
from . import io
from . import network
from . import runtime
def remove_script_to_evaluate_on_load(identifier: ScriptIdentifier) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Deprecated, please use removeScriptToEvaluateOnNewDocument instead.

    **EXPERIMENTAL**

    :param identifier:
    """
    params: T_JSON_DICT = dict()
    params['identifier'] = identifier.to_json()
    cmd_dict: T_JSON_DICT = {'method': 'Page.removeScriptToEvaluateOnLoad', 'params': params}
    json = (yield cmd_dict)