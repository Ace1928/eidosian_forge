from __future__ import annotations
import functools, uuid, os, operator
import typing as T
from . import backends
from .. import build
from .. import mesonlib
from .. import mlog
from ..mesonlib import MesonBugException, MesonException, OptionKey
def generate_project_tree(self) -> FileTreeEntry:
    tree_info = FileTreeEntry()
    for tname, t in self.build_targets.items():
        self.add_target_to_tree(tree_info, t)
    return tree_info