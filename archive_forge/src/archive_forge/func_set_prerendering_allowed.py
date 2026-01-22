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
def set_prerendering_allowed(is_allowed: bool) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Enable/disable prerendering manually.

    This command is a short-term solution for https://crbug.com/1440085.
    See https://docs.google.com/document/d/12HVmFxYj5Jc-eJr5OmWsa2bqTJsbgGLKI6ZIyx0_wpA
    for more details.

    TODO(https://crbug.com/1440085): Remove this once Puppeteer supports tab targets.

    **EXPERIMENTAL**

    :param is_allowed:
    """
    params: T_JSON_DICT = dict()
    params['isAllowed'] = is_allowed
    cmd_dict: T_JSON_DICT = {'method': 'Page.setPrerenderingAllowed', 'params': params}
    json = (yield cmd_dict)