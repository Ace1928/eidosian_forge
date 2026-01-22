import re
from io import BytesIO
from .lazy_import import lazy_import
from fnmatch import fnmatch
from breezy._termcolor import color_string, FG
from breezy import (
from . import controldir, errors, osutils
from . import revision as _mod_revision
from . import trace
from .revisionspec import RevisionSpec, RevisionSpec_revid, RevisionSpec_revno
def workingtree_grep(opts):
    revno = opts.print_revno = None
    tree, branch, relpath = controldir.ControlDir.open_containing_tree_or_branch('.')
    if not tree:
        msg = 'Cannot search working tree. Working tree not found.\nTo search for specific revision in history use the -r option.'
        raise errors.CommandError(msg)
    opts.outputter = _Outputter(opts)
    with tree.lock_read():
        for path in opts.path_list:
            if osutils.isdir(path):
                path_prefix = path
                dir_grep(tree, path, relpath, opts, revno, path_prefix)
            else:
                with open(path, 'rb') as f:
                    _file_grep(f.read(), path, opts, revno)