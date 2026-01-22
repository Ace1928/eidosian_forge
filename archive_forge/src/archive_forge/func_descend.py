from collections import deque
from ..lazy_import import lazy_import
from breezy.bzr import (
from .. import errors, lazy_regex, osutils, trace
from .static_tuple import StaticTuple
def descend(dir_ie, dir_path):
    kids = sorted(dir_ie.children.items())
    for name, ie in kids:
        child_path = osutils.pathjoin(dir_path, name)
        accum.append((child_path, ie))
        if ie.kind == 'directory':
            descend(ie, child_path)