from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import runtime

    Requests database names for given security origin.

    :param security_origin: Security origin.
    :returns: Database names for origin.
    