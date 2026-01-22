from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
def load_snapshot(tiles: typing.List[PictureTile]) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, SnapshotId]:
    """
    Returns the snapshot identifier.

    :param tiles: An array of tiles composing the snapshot.
    :returns: The id of the snapshot.
    """
    params: T_JSON_DICT = dict()
    params['tiles'] = [i.to_json() for i in tiles]
    cmd_dict: T_JSON_DICT = {'method': 'LayerTree.loadSnapshot', 'params': params}
    json = (yield cmd_dict)
    return SnapshotId.from_json(json['snapshotId'])