from __future__ import annotations
import abc
import datetime
import enum
from collections.abc import MutableMapping as _MutableMapping
from typing import (
from bson.binary import (
from bson.typings import _DocumentType
@abc.abstractproperty
def bson_type(self) -> Any:
    """The BSON type to be converted into our own type."""