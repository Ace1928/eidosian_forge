from __future__ import annotations
import functools, uuid, os, operator
import typing as T
from . import backends
from .. import build
from .. import mesonlib
from .. import mlog
from ..mesonlib import MesonBugException, MesonException, OptionKey
def generate_build_phase_map(self) -> None:
    for tname, t in self.build_targets.items():
        t.buildphasemap = {}
        t.buildphasemap[tname] = self.gen_id()
        t.buildphasemap['Frameworks'] = self.gen_id()
        t.buildphasemap['Resources'] = self.gen_id()
        t.buildphasemap['Sources'] = self.gen_id()