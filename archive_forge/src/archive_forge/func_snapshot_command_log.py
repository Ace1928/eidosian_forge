from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
def snapshot_command_log(snapshot_id: SnapshotId) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, typing.List[dict]]:
    """
    Replays the layer snapshot and returns canvas log.

    :param snapshot_id: The id of the layer snapshot.
    :returns: The array of canvas function calls.
    """
    params: T_JSON_DICT = dict()
    params['snapshotId'] = snapshot_id.to_json()
    cmd_dict: T_JSON_DICT = {'method': 'LayerTree.snapshotCommandLog', 'params': params}
    json = (yield cmd_dict)
    return [dict(i) for i in json['commandLog']]