import errno
import os
import shutil
from . import controldir, errors, ui
from .i18n import gettext
from .osutils import isdir
from .trace import note
from .workingtree import WorkingTree
def clean_tree(directory, unknown=False, ignored=False, detritus=False, dry_run=False, no_prompt=False):
    """Remove files in the specified classes from the tree"""
    tree = WorkingTree.open_containing(directory)[0]
    with tree.lock_read():
        deletables = list(iter_deletables(tree, unknown=unknown, ignored=ignored, detritus=detritus))
        deletables = _filter_out_nested_controldirs(deletables)
        if len(deletables) == 0:
            note(gettext('Nothing to delete.'))
            return 0
        if not no_prompt:
            for path, subp in deletables:
                ui.ui_factory.note(subp)
            prompt = gettext('Are you sure you wish to delete these')
            if not ui.ui_factory.get_boolean(prompt):
                ui.ui_factory.note(gettext('Canceled'))
                return 0
        delete_items(deletables, dry_run=dry_run)