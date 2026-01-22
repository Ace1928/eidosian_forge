from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import page
from . import target
def grant_permissions(permissions: typing.List[PermissionType], origin: typing.Optional[str]=None, browser_context_id: typing.Optional[BrowserContextID]=None) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Grant specific permissions to the given origin and reject all others.

    **EXPERIMENTAL**

    :param permissions:
    :param origin: *(Optional)* Origin the permission applies to, all origins if not specified.
    :param browser_context_id: *(Optional)* BrowserContext to override permissions. When omitted, default browser context is used.
    """
    params: T_JSON_DICT = dict()
    params['permissions'] = [i.to_json() for i in permissions]
    if origin is not None:
        params['origin'] = origin
    if browser_context_id is not None:
        params['browserContextId'] = browser_context_id.to_json()
    cmd_dict: T_JSON_DICT = {'method': 'Browser.grantPermissions', 'params': params}
    json = (yield cmd_dict)