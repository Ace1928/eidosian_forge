import contextlib
import errno
import os
import tempfile
import time
from stat import S_IEXEC, S_ISREG
from .. import (annotate, conflicts, controldir, errors, lock, multiparent,
from .. import revision as _mod_revision
from .. import trace
from .. import transport as _mod_transport
from .. import tree, ui, urlutils
from ..filters import ContentFilterContext, filtered_output_bytes
from ..i18n import gettext
from ..mutabletree import MutableTree
from ..progress import ProgressPhase
from ..transform import (ROOT_PARENT, FinalPaths, ImmortalLimbo,
from ..tree import find_previous_path
from . import inventory, inventorytree
from .conflicts import Conflict
def resolve_checkout(tt, conflicts, divert):
    new_conflicts = set()
    for c_type, conflict in ((c[0], c) for c in conflicts):
        if c_type != 'duplicate':
            raise AssertionError(c_type)
        if tt.new_contents(conflict[1]):
            new_file = conflict[1]
            old_file = conflict[2]
        else:
            new_file = conflict[2]
            old_file = conflict[1]
        final_parent = tt.final_parent(old_file)
        if new_file in divert:
            new_name = tt.final_name(old_file) + '.diverted'
            tt.adjust_path(new_name, final_parent, new_file)
            new_conflicts.add((c_type, 'Diverted to', new_file, old_file))
        else:
            new_name = tt.final_name(old_file) + '.moved'
            tt.adjust_path(new_name, final_parent, old_file)
            new_conflicts.add((c_type, 'Moved existing file to', old_file, new_file))
    return new_conflicts