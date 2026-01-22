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
def update_usage(self, timestep: int):
    """Expand self.live_range to include timestep"""
    self.live_range = LiveRange(min(timestep, self.live_range.begin), max(timestep, self.live_range.end))