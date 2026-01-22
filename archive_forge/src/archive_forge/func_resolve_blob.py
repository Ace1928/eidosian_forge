from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import runtime
def resolve_blob(object_id: runtime.RemoteObjectId) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, str]:
    """
    Return UUID of Blob object specified by a remote object id.

    :param object_id: Object id of a Blob object wrapper.
    :returns: UUID of the specified Blob.
    """
    params: T_JSON_DICT = dict()
    params['objectId'] = object_id.to_json()
    cmd_dict: T_JSON_DICT = {'method': 'IO.resolveBlob', 'params': params}
    json = (yield cmd_dict)
    return str(json['uuid'])