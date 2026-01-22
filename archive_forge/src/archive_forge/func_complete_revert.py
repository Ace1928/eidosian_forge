import os
from ... import config as _mod_config
from ... import osutils, ui
from ...bzr.generate_ids import gen_revision_id
from ...bzr.inventorytree import InventoryTreeChange
from ...errors import (BzrError, NoCommonAncestor, UnknownFormatError,
from ...graph import FrozenHeadsCache
from ...merge import Merger
from ...revision import NULL_REVISION
from ...trace import mutter
from ...transport import NoSuchFile
from ...tsort import topo_sort
from .maptree import MapTree, map_file_ids
def complete_revert(wt, newparents):
    """Simple helper that reverts to specified new parents and makes sure none
    of the extra files are left around.

    :param wt: Working tree to use for rebase
    :param newparents: New parents of the working tree
    """
    newtree = wt.branch.repository.revision_tree(newparents[0])
    delta = wt.changes_from(newtree)
    wt.branch.generate_revision_history(newparents[0])
    wt.set_parent_ids([r for r in newparents[:1] if r != NULL_REVISION])
    for change in delta.added:
        abs_path = wt.abspath(change.path[1])
        if osutils.lexists(abs_path):
            if osutils.isdir(abs_path):
                osutils.rmtree(abs_path)
            else:
                os.unlink(abs_path)
    wt.revert(None, old_tree=newtree, backups=False)
    assert not wt.changes_from(wt.basis_tree()).has_changed(), 'Rev changed'
    wt.set_parent_ids([r for r in newparents if r != NULL_REVISION])