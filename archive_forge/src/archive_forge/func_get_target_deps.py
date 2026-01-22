from __future__ import annotations
import copy
import itertools
import os
import xml.dom.minidom
import xml.etree.ElementTree as ET
import uuid
import typing as T
from pathlib import Path, PurePath, PureWindowsPath
import re
from collections import Counter
from . import backends
from .. import build
from .. import mlog
from .. import compilers
from .. import mesonlib
from ..mesonlib import (
from ..environment import Environment, build_filename
from .. import coredata
def get_target_deps(self, t: T.Dict[T.Any, build.Target], recursive=False):
    all_deps: T.Dict[str, build.Target] = {}
    for target in t.values():
        if isinstance(target, build.CustomTarget):
            for d in target.get_target_dependencies():
                if isinstance(d, build.Target):
                    all_deps[d.get_id()] = d
        elif isinstance(target, build.RunTarget):
            for d in target.get_dependencies():
                all_deps[d.get_id()] = d
        elif isinstance(target, build.BuildTarget):
            for ldep in target.link_targets:
                if isinstance(ldep, build.CustomTargetIndex):
                    all_deps[ldep.get_id()] = ldep.target
                else:
                    all_deps[ldep.get_id()] = ldep
            for ldep in target.link_whole_targets:
                if isinstance(ldep, build.CustomTargetIndex):
                    all_deps[ldep.get_id()] = ldep.target
                else:
                    all_deps[ldep.get_id()] = ldep
            for ldep in target.link_depends:
                if isinstance(ldep, build.CustomTargetIndex):
                    all_deps[ldep.get_id()] = ldep.target
                elif isinstance(ldep, File):
                    pass
                else:
                    all_deps[ldep.get_id()] = ldep
            for obj_id, objdep in self.get_obj_target_deps(target.objects):
                all_deps[obj_id] = objdep
        else:
            raise MesonException(f'Unknown target type for target {target}')
        for gendep in target.get_generated_sources():
            if isinstance(gendep, build.CustomTarget):
                all_deps[gendep.get_id()] = gendep
            elif isinstance(gendep, build.CustomTargetIndex):
                all_deps[gendep.target.get_id()] = gendep.target
            else:
                generator = gendep.get_generator()
                gen_exe = generator.get_exe()
                if isinstance(gen_exe, build.Executable):
                    all_deps[gen_exe.get_id()] = gen_exe
                for d in itertools.chain(generator.depends, gendep.depends):
                    if isinstance(d, build.CustomTargetIndex):
                        all_deps[d.get_id()] = d.target
                    elif isinstance(d, build.Target):
                        all_deps[d.get_id()] = d
    if not t or not recursive:
        return all_deps
    ret = self.get_target_deps(all_deps, recursive)
    ret.update(all_deps)
    return ret