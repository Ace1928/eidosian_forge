from __future__ import annotations
from enum import IntEnum
from functools import partial
from typing import Any, Callable, List, Mapping, Optional, Sequence, TypeVar
import attrs
from attrs import define, field
from fontTools.ufoLib import UFOReader
from ufoLib2.objects.guideline import Guideline
from ufoLib2.objects.misc import AttrDictMixin
from ufoLib2.serde import serde
from .woff import (
@openTypeNameRecords.setter
def openTypeNameRecords(self, value: list[NameRecord] | None) -> None:
    self._openTypeNameRecords = _convert_name_records(value)