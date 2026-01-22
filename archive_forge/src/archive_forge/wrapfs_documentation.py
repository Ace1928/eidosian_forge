from __future__ import unicode_literals
import typing
import six
from . import errors
from .base import FS
from .copy import copy_dir, copy_file
from .error_tools import unwrap_errors
from .info import Info
from .path import abspath, join, normpath
Get the proxied filesystem.

        This method should return a filesystem for methods not
        associated with a path, e.g. `~fs.base.FS.getmeta`.

        