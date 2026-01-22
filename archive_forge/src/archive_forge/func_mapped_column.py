from __future__ import annotations
import sys
from . import config
from . import exclusions
from .. import event
from .. import schema
from .. import types as sqltypes
from ..orm import mapped_column as _orm_mapped_column
from ..util import OrderedDict
def mapped_column(*args, **kw):
    """An orm.mapped_column wrapper/hook for dialect-specific tweaks."""
    return _schema_column(_orm_mapped_column, args, kw)