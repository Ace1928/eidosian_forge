from __future__ import annotations
import functools, uuid, os, operator
import typing as T
from . import backends
from .. import build
from .. import mesonlib
from .. import mlog
from ..mesonlib import MesonBugException, MesonException, OptionKey
def generate_generator_target_shell_build_phases(self, objects_dict: PbxDict) -> None:
    for tname, t in self.build_targets.items():
        generator_id = 0
        for genlist in t.generated:
            if isinstance(genlist, build.GeneratedList):
                self.generate_single_generator_phase(tname, t, genlist, generator_id, objects_dict)
                generator_id += 1
    for tname, t in self.custom_targets.items():
        generator_id = 0
        for genlist in t.sources:
            if isinstance(genlist, build.GeneratedList):
                self.generate_single_generator_phase(tname, t, genlist, generator_id, objects_dict)
                generator_id += 1