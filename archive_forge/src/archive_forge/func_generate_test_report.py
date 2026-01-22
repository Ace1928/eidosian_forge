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
def generate_test_report(message: str, group: typing.Optional[str]=None) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Generates a report for testing.

    **EXPERIMENTAL**

    :param message: Message to be displayed in the report.
    :param group: *(Optional)* Specifies the endpoint group to deliver the report to.
    """
    params: T_JSON_DICT = dict()
    params['message'] = message
    if group is not None:
        params['group'] = group
    cmd_dict: T_JSON_DICT = {'method': 'Page.generateTestReport', 'params': params}
    json = (yield cmd_dict)