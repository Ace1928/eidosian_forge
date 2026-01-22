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
def final_file_id(self, trans_id):
    """Determine the file id after any changes are applied, or None.

        None indicates that the file will not be versioned after changes are
        applied.
        """
    try:
        return self._new_id[trans_id]
    except KeyError:
        if trans_id in self._removed_id:
            return None
    return self.tree_file_id(trans_id)