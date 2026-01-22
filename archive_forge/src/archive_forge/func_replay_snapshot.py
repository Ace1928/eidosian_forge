from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
def replay_snapshot(snapshot_id: SnapshotId, from_step: typing.Optional[int]=None, to_step: typing.Optional[int]=None, scale: typing.Optional[float]=None) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, str]:
    """
    Replays the layer snapshot and returns the resulting bitmap.

    :param snapshot_id: The id of the layer snapshot.
    :param from_step: *(Optional)* The first step to replay from (replay from the very start if not specified).
    :param to_step: *(Optional)* The last step to replay to (replay till the end if not specified).
    :param scale: *(Optional)* The scale to apply while replaying (defaults to 1).
    :returns: A data: URL for resulting image.
    """
    params: T_JSON_DICT = dict()
    params['snapshotId'] = snapshot_id.to_json()
    if from_step is not None:
        params['fromStep'] = from_step
    if to_step is not None:
        params['toStep'] = to_step
    if scale is not None:
        params['scale'] = scale
    cmd_dict: T_JSON_DICT = {'method': 'LayerTree.replaySnapshot', 'params': params}
    json = (yield cmd_dict)
    return str(json['dataURL'])