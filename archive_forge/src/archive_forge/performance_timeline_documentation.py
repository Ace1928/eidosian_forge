from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import network
from . import page

    Sent when a performance timeline event is added. See reportPerformanceTimeline method.
    