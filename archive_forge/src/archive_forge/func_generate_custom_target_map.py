from __future__ import annotations
import functools, uuid, os, operator
import typing as T
from . import backends
from .. import build
from .. import mesonlib
from .. import mlog
from ..mesonlib import MesonBugException, MesonException, OptionKey
def generate_custom_target_map(self) -> None:
    self.shell_targets = {}
    self.custom_target_output_buildfile = {}
    self.custom_target_output_fileref = {}
    for tname, t in self.custom_targets.items():
        self.shell_targets[tname] = self.gen_id()
        if not isinstance(t, build.CustomTarget):
            continue
        srcs, ofilenames, cmd = self.eval_custom_target_command(t)
        for o in ofilenames:
            self.custom_target_output_buildfile[o] = self.gen_id()
            self.custom_target_output_fileref[o] = self.gen_id()