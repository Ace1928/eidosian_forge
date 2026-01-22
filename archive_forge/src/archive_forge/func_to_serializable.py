from __future__ import annotations
import logging # isort:skip
from typing import TYPE_CHECKING, Any, Union
from ...util.dataclasses import NotRequired, Unspecified, dataclass
from ..serialization import (
def to_serializable(self, serializer: Serializer) -> AnyRep:
    return serializer.encode_struct(type='expr', expr=self.expr, transform=self.transform, units=self.units)