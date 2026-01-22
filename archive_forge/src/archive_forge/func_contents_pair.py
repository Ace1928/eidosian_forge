import contextlib
import tempfile
from typing import Type
from .lazy_import import lazy_import
import patiencediff
from breezy import (
from breezy.bzr import (
from breezy.i18n import gettext
from . import decorators, errors, hooks, osutils, registry
from . import revision as _mod_revision
from . import trace, transform
from . import transport as _mod_transport
from . import tree as _mod_tree
def contents_pair(tree, path):
    if path is None:
        return (None, None)
    try:
        kind = tree.kind(path)
    except _mod_transport.NoSuchFile:
        return (None, None)
    if kind == 'file':
        contents = tree.get_file_sha1(path)
    elif kind == 'symlink':
        contents = tree.get_symlink_target(path)
    else:
        contents = None
    return (kind, contents)