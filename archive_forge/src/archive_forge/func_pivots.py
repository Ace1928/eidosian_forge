from __future__ import annotations
import itertools
import logging
import typing as t
from collections import defaultdict
from enum import Enum, auto
from sqlglot import exp
from sqlglot.errors import OptimizeError
from sqlglot.helper import ensure_collection, find_new_name, seq_get
@property
def pivots(self):
    if not self._pivots:
        self._pivots = [pivot for _, node in self.references for pivot in node.args.get('pivots') or []]
    return self._pivots