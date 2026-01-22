from __future__ import annotations
import gc
import logging
import os
import os.path as osp
from pathlib import Path
import re
import shlex
import warnings
import gitdb
from gitdb.db.loose import LooseObjectDB
from gitdb.exc import BadObject
from git.cmd import Git, handle_process_output
from git.compat import defenc, safe_decode
from git.config import GitConfigParser
from git.db import GitCmdObjectDB
from git.exc import (
from git.index import IndexFile
from git.objects import Submodule, RootModule, Commit
from git.refs import HEAD, Head, Reference, TagReference
from git.remote import Remote, add_progress, to_progress_instance
from git.util import (
from .fun import (
from git.types import (
from typing import (
from git.types import ConfigLevels_Tup, TypedDict
def clone(self, path: PathLike, progress: Optional[CallableProgress]=None, multi_options: Optional[List[str]]=None, allow_unsafe_protocols: bool=False, allow_unsafe_options: bool=False, **kwargs: Any) -> 'Repo':
    """Create a clone from this repository.

        :param path: The full path of the new repo (traditionally ends with
            ``./<name>.git``).
        :param progress: See :meth:`git.remote.Remote.push`.
        :param multi_options: A list of Clone options that can be provided multiple times.
            One option per list item which is passed exactly as specified to clone.
            For example: ['--config core.filemode=false', '--config core.ignorecase',
            '--recurse-submodule=repo1_path', '--recurse-submodule=repo2_path']
        :param allow_unsafe_protocols: Allow unsafe protocols to be used, like ext.
        :param allow_unsafe_options: Allow unsafe options to be used, like --upload-pack.
        :param kwargs:
            * odbt = ObjectDatabase Type, allowing to determine the object database
              implementation used by the returned Repo instance.
            * All remaining keyword arguments are given to the git-clone command.

        :return: :class:`Repo` (the newly cloned repo)
        """
    return self._clone(self.git, self.common_dir, path, type(self.odb), progress, multi_options, allow_unsafe_protocols=allow_unsafe_protocols, allow_unsafe_options=allow_unsafe_options, **kwargs)