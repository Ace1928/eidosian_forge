from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import page
def get_frames_with_manifests() -> typing.Generator[T_JSON_DICT, T_JSON_DICT, typing.List[FrameWithManifest]]:
    """
    Returns array of frame identifiers with manifest urls for each frame containing a document
    associated with some application cache.

    :returns: Array of frame identifiers with manifest urls for each frame containing a document associated with some application cache.
    """
    cmd_dict: T_JSON_DICT = {'method': 'ApplicationCache.getFramesWithManifests'}
    json = (yield cmd_dict)
    return [FrameWithManifest.from_json(i) for i in json['frameIds']]