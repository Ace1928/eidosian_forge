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
def config_reader(self) -> SectionConstraint[SubmoduleConfigParser]:
    """
        :return: ConfigReader instance which allows you to query the configuration
            values of this submodule, as provided by the .gitmodules file.

        :note: The config reader will actually read the data directly from the
            repository and thus does not need nor care about your working tree.

        :note: Should be cached by the caller and only kept as long as needed.

        :raise IOError: If the .gitmodules file/blob could not be read.
        """
    return self._config_parser_constrained(read_only=True)