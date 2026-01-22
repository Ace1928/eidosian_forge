import errno
import os
import shutil
from . import controldir, errors, ui
from .i18n import gettext
from .osutils import isdir
from .trace import note
from .workingtree import WorkingTree
def iter_deletables(tree, unknown=False, ignored=False, detritus=False):
    """Iterate through files that may be deleted"""
    for subp in tree.extras():
        if detritus and is_detritus(subp):
            yield (tree.abspath(subp), subp)
            continue
        if tree.is_ignored(subp):
            if ignored:
                yield (tree.abspath(subp), subp)
        elif unknown:
            yield (tree.abspath(subp), subp)