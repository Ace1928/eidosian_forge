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
def submodule_update(self, *args: Any, **kwargs: Any) -> Iterator[Submodule]:
    """Update the submodules, keeping the repository consistent as it will
        take the previous state into consideration.

        :note: For more information, please see the documentation of
            :meth:`RootModule.update <git.objects.submodule.root.RootModule.update>`.
        """
    return RootModule(self).update(*args, **kwargs)