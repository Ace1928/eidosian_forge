from __future__ import absolute_import, print_function, unicode_literals
import typing
import abc
import hashlib
import itertools
import os
import six
import threading
import time
import warnings
from contextlib import closing
from functools import partial, wraps
from . import copy, errors, fsencode, iotools, tools, walk, wildcard
from .copy import copy_modified_time
from .glob import BoundGlobber
from .mode import validate_open_mode
from .path import abspath, join, normpath
from .time import datetime_to_epoch
from .walk import Walker
def getbasic(self, path):
    """Get the *basic* resource info.

        This method is shorthand for the following::

            fs.getinfo(path, namespaces=['basic'])

        Arguments:
            path (str): A path on the filesystem.

        Returns:
            ~fs.info.Info: Resource information object for ``path``.

        Note:
            .. deprecated:: 2.4.13
                Please use `~FS.getinfo` directly, which is
                required to always return the *basic* namespace.

        """
    warnings.warn("method 'getbasic' has been deprecated, please use 'getinfo'", DeprecationWarning)
    return self.getinfo(path, namespaces=['basic'])