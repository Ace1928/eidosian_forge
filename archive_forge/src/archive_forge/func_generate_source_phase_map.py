from __future__ import annotations
import functools, uuid, os, operator
import typing as T
from . import backends
from .. import build
from .. import mesonlib
from .. import mlog
from ..mesonlib import MesonBugException, MesonException, OptionKey
def generate_source_phase_map(self) -> None:
    self.source_phase = {}
    for t in self.build_targets:
        self.source_phase[t] = self.gen_id()