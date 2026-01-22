from __future__ import annotations
import functools, uuid, os, operator
import typing as T
from . import backends
from .. import build
from .. import mesonlib
from .. import mlog
from ..mesonlib import MesonBugException, MesonException, OptionKey
def generate_target_file_maps_impl(self, targets) -> None:
    for tname, t in targets.items():
        for s in t.sources:
            if isinstance(s, mesonlib.File):
                s = os.path.join(s.subdir, s.fname)
            if not isinstance(s, str):
                continue
            k = (tname, s)
            assert k not in self.buildfile_ids
            self.buildfile_ids[k] = self.gen_id()
            assert k not in self.fileref_ids
            self.fileref_ids[k] = self.gen_id()
        if not hasattr(t, 'objects'):
            continue
        for o in t.objects:
            if isinstance(o, build.ExtractedObjects):
                continue
            if isinstance(o, mesonlib.File):
                o = os.path.join(o.subdir, o.fname)
            if isinstance(o, str):
                o = os.path.join(t.subdir, o)
                k = (tname, o)
                assert k not in self.buildfile_ids
                self.buildfile_ids[k] = self.gen_id()
                assert k not in self.fileref_ids
                self.fileref_ids[k] = self.gen_id()
            else:
                raise RuntimeError('Unknown input type ' + str(o))
        for e in t.extra_files:
            if isinstance(e, mesonlib.File):
                e = os.path.join(e.subdir, e.fname)
            if isinstance(e, str):
                e = os.path.join(t.subdir, e)
                k = (tname, e)
                assert k not in self.buildfile_ids
                self.buildfile_ids[k] = self.gen_id()
                assert k not in self.fileref_ids
                self.fileref_ids[k] = self.gen_id()