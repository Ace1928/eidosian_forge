from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import page
from . import runtime
def set_node_stack_traces_enabled(enable: bool) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Sets if stack traces should be captured for Nodes. See ``Node.getNodeStackTraces``. Default is disabled.

    **EXPERIMENTAL**

    :param enable: Enable or disable.
    """
    params: T_JSON_DICT = dict()
    params['enable'] = enable
    cmd_dict: T_JSON_DICT = {'method': 'DOM.setNodeStackTracesEnabled', 'params': params}
    json = (yield cmd_dict)