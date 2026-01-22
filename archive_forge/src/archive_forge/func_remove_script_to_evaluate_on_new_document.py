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
def remove_script_to_evaluate_on_new_document(identifier: ScriptIdentifier) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Removes given script from the list.

    :param identifier:
    """
    params: T_JSON_DICT = dict()
    params['identifier'] = identifier.to_json()
    cmd_dict: T_JSON_DICT = {'method': 'Page.removeScriptToEvaluateOnNewDocument', 'params': params}
    json = (yield cmd_dict)