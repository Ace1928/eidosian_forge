from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import browser
from . import page
def get_browser_contexts() -> typing.Generator[T_JSON_DICT, T_JSON_DICT, typing.List[browser.BrowserContextID]]:
    """
    Returns all browser contexts created with ``Target.createBrowserContext`` method.

    :returns: An array of browser context ids.
    """
    cmd_dict: T_JSON_DICT = {'method': 'Target.getBrowserContexts'}
    json = (yield cmd_dict)
    return [browser.BrowserContextID.from_json(i) for i in json['browserContextIds']]