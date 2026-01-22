from __future__ import annotations
import collections
import re
import typing
from typing import Any
from typing import Dict
from typing import Optional
import warnings
import weakref
from . import config
from .util import decorator
from .util import gc_collect
from .. import event
from .. import pool
from ..util import await_only
from ..util.typing import Literal
def rollback_all(self):
    for rec in list(self.proxy_refs):
        if rec is not None and rec.is_valid:
            self._safe(rec.rollback)