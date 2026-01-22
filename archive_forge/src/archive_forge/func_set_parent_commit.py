import gc
from io import BytesIO
import logging
import os
import os.path as osp
import stat
import uuid
import git
from git.cmd import Git
from git.compat import defenc
from git.config import GitConfigParser, SectionConstraint, cp
from git.exc import (
from git.objects.base import IndexObject, Object
from git.objects.util import TraversableIterableObj
from git.util import (
from .util import (
from typing import Callable, Dict, Mapping, Sequence, TYPE_CHECKING, cast
from typing import Any, Iterator, Union
from git.types import Commit_ish, Literal, PathLike, TBD
def set_parent_commit(self, commit: Union[Commit_ish, None], check: bool=True) -> 'Submodule':
    """Set this instance to use the given commit whose tree is supposed to
        contain the .gitmodules blob.

        :param commit:
            Commit-ish reference pointing at the root_tree, or None to always point to
            the most recent commit
        :param check:
            If True, relatively expensive checks will be performed to verify
            validity of the submodule.
        :raise ValueError: If the commit's tree didn't contain the .gitmodules blob.
        :raise ValueError:
            If the parent commit didn't store this submodule under the current path.
        :return: self
        """
    if commit is None:
        self._parent_commit = None
        return self
    pcommit = self.repo.commit(commit)
    pctree = pcommit.tree
    if self.k_modules_file not in pctree:
        raise ValueError('Tree of commit %s did not contain the %s file' % (commit, self.k_modules_file))
    prev_pc = self._parent_commit
    self._parent_commit = pcommit
    if check:
        parser = self._config_parser(self.repo, self._parent_commit, read_only=True)
        if not parser.has_section(sm_section(self.name)):
            self._parent_commit = prev_pc
            raise ValueError('Submodule at path %r did not exist in parent commit %s' % (self.path, commit))
    try:
        self.binsha = pctree[str(self.path)].binsha
    except KeyError:
        self.binsha = self.NULL_BIN_SHA
    self._clear_cache()
    return self