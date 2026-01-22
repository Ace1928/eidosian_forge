from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import browser
from . import page
def set_remote_locations(locations: typing.List[RemoteLocation]) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Enables target discovery for the specified locations, when ``setDiscoverTargets`` was set to
    ``true``.

    **EXPERIMENTAL**

    :param locations: List of remote locations.
    """
    params: T_JSON_DICT = dict()
    params['locations'] = [i.to_json() for i in locations]
    cmd_dict: T_JSON_DICT = {'method': 'Target.setRemoteLocations', 'params': params}
    json = (yield cmd_dict)