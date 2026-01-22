from __future__ import annotations
import collections
import dataclasses
import itertools
import pprint
from typing import Any, Dict, Iterable, List, Optional, Protocol
import sympy
import torch
from .. import config, ir
from ..utils import cache_on_self, CachedMethod, IndentedBuffer
from ..virtualized import V
from .wrapper import (
def mark_first_last_usage(self, lines):
    """
        Populate the AllocFromPoolLine.is_first_pool_usage and
        DeallocFromPoolLine.is_last_pool_usage fields so that pools
        are created/destroyed.
        """
    seen = set()
    for line in lines:
        if isinstance(line, AllocFromPoolLine):
            assert line.group.allocation
            pool = line.group.allocation.pool
            assert pool is not None
            if pool not in seen:
                line.is_first_pool_usage = True
                seen.add(pool)
    seen = set()
    for line in reversed(lines):
        if isinstance(line, DeallocFromPoolLine):
            assert line.group.allocation
            pool = line.group.allocation.pool
            assert pool is not None
            if pool not in seen:
                line.is_last_pool_usage = pool.root.get_live_ranges().end <= line.timestep
                seen.add(pool)