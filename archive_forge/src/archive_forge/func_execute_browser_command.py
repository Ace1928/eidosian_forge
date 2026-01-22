from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import page
from . import target
def execute_browser_command(command_id: BrowserCommandId) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Invoke custom browser commands used by telemetry.

    **EXPERIMENTAL**

    :param command_id:
    """
    params: T_JSON_DICT = dict()
    params['commandId'] = command_id.to_json()
    cmd_dict: T_JSON_DICT = {'method': 'Browser.executeBrowserCommand', 'params': params}
    json = (yield cmd_dict)