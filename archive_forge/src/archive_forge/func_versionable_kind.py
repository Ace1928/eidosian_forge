from collections import deque
from ..lazy_import import lazy_import
from breezy.bzr import (
from .. import errors, lazy_regex, osutils, trace
from .static_tuple import StaticTuple
@staticmethod
def versionable_kind(kind):
    return kind in ('file', 'directory', 'symlink', 'tree-reference')