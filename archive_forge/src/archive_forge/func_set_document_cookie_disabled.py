from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import network
from . import page
def set_document_cookie_disabled(disabled: bool) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """


    **EXPERIMENTAL**

    :param disabled: Whether document.coookie API should be disabled.
    """
    params: T_JSON_DICT = dict()
    params['disabled'] = disabled
    cmd_dict: T_JSON_DICT = {'method': 'Emulation.setDocumentCookieDisabled', 'params': params}
    json = (yield cmd_dict)