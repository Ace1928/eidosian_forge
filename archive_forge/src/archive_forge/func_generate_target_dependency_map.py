from __future__ import annotations
import functools, uuid, os, operator
import typing as T
from . import backends
from .. import build
from .. import mesonlib
from .. import mlog
from ..mesonlib import MesonBugException, MesonException, OptionKey
def generate_target_dependency_map(self) -> None:
    self.target_dependency_map = {}
    for tname, t in self.build_targets.items():
        for target in t.link_targets:
            if isinstance(target, build.CustomTargetIndex):
                k = (tname, target.target.get_basename())
                if k in self.target_dependency_map:
                    continue
            else:
                k = (tname, target.get_basename())
                assert k not in self.target_dependency_map
            self.target_dependency_map[k] = self.gen_id()
    for tname, t in self.custom_targets.items():
        k = tname
        assert k not in self.target_dependency_map
        self.target_dependency_map[k] = self.gen_id()