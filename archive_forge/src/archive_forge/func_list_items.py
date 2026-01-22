from abc import abstractmethod
import contextlib
from functools import wraps
import getpass
import logging
import os
import os.path as osp
import pathlib
import platform
import re
import shutil
import stat
import subprocess
import sys
import time
from urllib.parse import urlsplit, urlunsplit
import warnings
from typing import (
from .types import (
from gitdb.util import (  # noqa: F401  # @IgnorePep8
@classmethod
def list_items(cls, repo: 'Repo', *args: Any, **kwargs: Any) -> Any:
    """Deprecated, use :class:`IterableObj` instead.

        Find (all) items of this type and collect them into a list.

        See :meth:`IterableObj.list_items` for details on usage.

        :return: list(Item,...) list of item instances
        """
    out_list: Any = IterableList(cls._id_attribute_)
    out_list.extend(cls.iter_items(repo, *args, **kwargs))
    return out_list