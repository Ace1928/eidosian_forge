from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import browser
from . import page
def dispose_browser_context(browser_context_id: browser.BrowserContextID) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Deletes a BrowserContext. All the belonging pages will be closed without calling their
    beforeunload hooks.

    :param browser_context_id:
    """
    params: T_JSON_DICT = dict()
    params['browserContextId'] = browser_context_id.to_json()
    cmd_dict: T_JSON_DICT = {'method': 'Target.disposeBrowserContext', 'params': params}
    json = (yield cmd_dict)