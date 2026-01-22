from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
def release_object(object_id: RemoteObjectId) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Releases remote object with given id.

    :param object_id: Identifier of the object to release.
    """
    params: T_JSON_DICT = dict()
    params['objectId'] = object_id.to_json()
    cmd_dict: T_JSON_DICT = {'method': 'Runtime.releaseObject', 'params': params}
    json = (yield cmd_dict)