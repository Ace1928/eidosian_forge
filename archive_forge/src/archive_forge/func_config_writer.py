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
@unbare_repo
def config_writer(self, index: Union['IndexFile', None]=None, write: bool=True) -> SectionConstraint['SubmoduleConfigParser']:
    """
        :return: A config writer instance allowing you to read and write the data
            belonging to this submodule into the .gitmodules file.

        :param index: If not None, an IndexFile instance which should be written.
            Defaults to the index of the Submodule's parent repository.
        :param write: If True, the index will be written each time a configuration
            value changes.

        :note: The parameters allow for a more efficient writing of the index,
            as you can pass in a modified index on your own, prevent automatic writing,
            and write yourself once the whole operation is complete.

        :raise ValueError: If trying to get a writer on a parent_commit which does not
            match the current head commit.
        :raise IOError: If the .gitmodules file/blob could not be read
        """
    writer = self._config_parser_constrained(read_only=False)
    if index is not None:
        writer.config._index = index
    writer.config._auto_write = write
    return writer