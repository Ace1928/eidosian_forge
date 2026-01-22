from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import network
from . import page
from . import runtime
def check_contrast(report_aaa: typing.Optional[bool]=None) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Runs the contrast check for the target page. Found issues are reported
    using Audits.issueAdded event.

    :param report_aaa: *(Optional)* Whether to report WCAG AAA level issues. Default is false.
    """
    params: T_JSON_DICT = dict()
    if report_aaa is not None:
        params['reportAAA'] = report_aaa
    cmd_dict: T_JSON_DICT = {'method': 'Audits.checkContrast', 'params': params}
    json = (yield cmd_dict)