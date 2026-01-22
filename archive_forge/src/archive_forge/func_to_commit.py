from __future__ import annotations
import os
import stat
from pathlib import Path
from string import digits
from git.exc import WorkTreeRepositoryUnsupported
from git.objects import Object
from git.refs import SymbolicReference
from git.util import hex_to_bin, bin_to_hex, cygpath
from gitdb.exc import (
import os.path as osp
from git.cmd import Git
from typing import Union, Optional, cast, TYPE_CHECKING
from git.types import Commit_ish
def to_commit(obj: Object) -> Union['Commit', 'TagObject']:
    """Convert the given object to a commit if possible and return it"""
    if obj.type == 'tag':
        obj = deref_tag(obj)
    if obj.type != 'commit':
        raise ValueError('Cannot convert object %r to type commit' % obj)
    return obj