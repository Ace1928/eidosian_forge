import datetime
import re
from subprocess import Popen, PIPE
from gitdb import IStream
from git.util import hex_to_bin, Actor, Stats, finalize_process
from git.diff import Diffable
from git.cmd import Git
from .tree import Tree
from . import base
from .util import (
from time import time, daylight, altzone, timezone, localtime
import os
from io import BytesIO
import logging
from collections import defaultdict
from typing import (
from git.types import PathLike, Literal
def iter_parents(self, paths: Union[PathLike, Sequence[PathLike]]='', **kwargs: Any) -> Iterator['Commit']:
    """Iterate _all_ parents of this commit.

        :param paths:
            Optional path or list of paths limiting the Commits to those that
            contain at least one of the paths
        :param kwargs: All arguments allowed by git-rev-list
        :return: Iterator yielding Commit objects which are parents of self
        """
    skip = kwargs.get('skip', 1)
    if skip == 0:
        skip = 1
    kwargs['skip'] = skip
    return self.iter_items(self.repo, self, paths, **kwargs)