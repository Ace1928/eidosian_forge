import itertools
import os
import struct
from copy import copy
from io import BytesIO
from typing import Any, Tuple
from zlib import adler32
from ..lazy_import import lazy_import
import fastbencode as bencode
from breezy import (
from breezy.bzr import (
from .. import errors
from .. import graph as _mod_graph
from .. import osutils
from .. import transport as _mod_transport
from ..registry import Registry
from ..textmerge import TextMerge
from . import index
def plan_lca_merge(self, ver_a, ver_b, base=None):
    from ..merge import _PlanLCAMerge
    graph = _mod_graph.Graph(self)
    new_plan = _PlanLCAMerge(ver_a, ver_b, self, (self._file_id,), graph).plan_merge()
    if base is None:
        return new_plan
    old_plan = _PlanLCAMerge(ver_a, base, self, (self._file_id,), graph).plan_merge()
    return _PlanLCAMerge._subtract_plans(list(old_plan), list(new_plan))