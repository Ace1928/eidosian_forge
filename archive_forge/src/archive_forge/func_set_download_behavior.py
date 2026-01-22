from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import page
from . import target
def set_download_behavior(behavior: str, browser_context_id: typing.Optional[BrowserContextID]=None, download_path: typing.Optional[str]=None, events_enabled: typing.Optional[bool]=None) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Set the behavior when downloading a file.

    **EXPERIMENTAL**

    :param behavior: Whether to allow all or deny all download requests, or use default Chrome behavior if available (otherwise deny). ``allowAndName`` allows download and names files according to their dowmload guids.
    :param browser_context_id: *(Optional)* BrowserContext to set download behavior. When omitted, default browser context is used.
    :param download_path: *(Optional)* The default path to save downloaded files to. This is required if behavior is set to 'allow' or 'allowAndName'.
    :param events_enabled: *(Optional)* Whether to emit download events (defaults to false).
    """
    params: T_JSON_DICT = dict()
    params['behavior'] = behavior
    if browser_context_id is not None:
        params['browserContextId'] = browser_context_id.to_json()
    if download_path is not None:
        params['downloadPath'] = download_path
    if events_enabled is not None:
        params['eventsEnabled'] = events_enabled
    cmd_dict: T_JSON_DICT = {'method': 'Browser.setDownloadBehavior', 'params': params}
    json = (yield cmd_dict)