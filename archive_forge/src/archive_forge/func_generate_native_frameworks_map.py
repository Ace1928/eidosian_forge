from __future__ import annotations
import functools, uuid, os, operator
import typing as T
from . import backends
from .. import build
from .. import mesonlib
from .. import mlog
from ..mesonlib import MesonBugException, MesonException, OptionKey
def generate_native_frameworks_map(self) -> None:
    self.native_frameworks = {}
    self.native_frameworks_fileref = {}
    for t in self.build_targets.values():
        for dep in t.get_external_deps():
            if dep.name == 'appleframeworks':
                for f in dep.frameworks:
                    self.native_frameworks[f] = self.gen_id()
                    self.native_frameworks_fileref[f] = self.gen_id()