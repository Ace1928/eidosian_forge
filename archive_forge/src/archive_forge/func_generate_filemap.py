from __future__ import annotations
import functools, uuid, os, operator
import typing as T
from . import backends
from .. import build
from .. import mesonlib
from .. import mlog
from ..mesonlib import MesonBugException, MesonException, OptionKey
def generate_filemap(self) -> None:
    self.filemap = {}
    self.target_filemap = {}
    for name, t in self.build_targets.items():
        for s in t.sources:
            if isinstance(s, mesonlib.File):
                s = os.path.join(s.subdir, s.fname)
                self.filemap[s] = self.gen_id()
        for o in t.objects:
            if isinstance(o, str):
                o = os.path.join(t.subdir, o)
                self.filemap[o] = self.gen_id()
        for e in t.extra_files:
            if isinstance(e, mesonlib.File):
                e = os.path.join(e.subdir, e.fname)
                self.filemap[e] = self.gen_id()
            else:
                e = os.path.join(t.subdir, e)
                self.filemap[e] = self.gen_id()
        self.target_filemap[name] = self.gen_id()