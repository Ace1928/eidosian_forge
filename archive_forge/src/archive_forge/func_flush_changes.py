import abc
import configparser as cp
import fnmatch
from functools import wraps
import inspect
from io import BufferedReader, IOBase
import logging
import os
import os.path as osp
import re
import sys
from git.compat import defenc, force_text
from git.util import LockFile
from typing import (
from git.types import Lit_config_levels, ConfigLevels_Tup, PathLike, assert_never, _T
def flush_changes(self: 'GitConfigParser', *args: Any, **kwargs: Any) -> _T:
    rval = non_const_func(self, *args, **kwargs)
    self._dirty = True
    self.write()
    return rval