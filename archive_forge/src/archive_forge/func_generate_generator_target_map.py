from __future__ import annotations
import functools, uuid, os, operator
import typing as T
from . import backends
from .. import build
from .. import mesonlib
from .. import mlog
from ..mesonlib import MesonBugException, MesonException, OptionKey
def generate_generator_target_map(self) -> None:
    self.generator_fileref_ids = {}
    self.generator_buildfile_ids = {}
    for tname, t in self.build_targets.items():
        generator_id = 0
        for genlist in t.generated:
            if not isinstance(genlist, build.GeneratedList):
                continue
            self.gen_single_target_map(genlist, tname, t, generator_id)
            generator_id += 1
    for tname, t in self.custom_targets.items():
        generator_id = 0
        for genlist in t.sources:
            if not isinstance(genlist, build.GeneratedList):
                continue
            self.gen_single_target_map(genlist, tname, t, generator_id)
            generator_id += 1