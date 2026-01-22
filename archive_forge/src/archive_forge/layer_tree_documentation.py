from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom

    Replays the layer snapshot and returns canvas log.

    :param snapshot_id: The id of the layer snapshot.
    :returns: The array of canvas function calls.
    