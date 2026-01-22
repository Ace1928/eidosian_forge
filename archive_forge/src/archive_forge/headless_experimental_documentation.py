from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing

    Issued when the target starts or stops needing BeginFrames.
    Deprecated. Issue beginFrame unconditionally instead and use result from
    beginFrame to detect whether the frames were suppressed.
    