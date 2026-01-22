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
def determine_base(self, oldrevid, oldparents, newrevid, newparents):
    """Determine the base for replaying a revision using merge.

        :param oldrevid: Revid of old revision.
        :param oldparents: List of old parents revids.
        :param newrevid: Revid of new revision.
        :param newparents: List of new parents revids.
        :return: Revision id of the new new revision.
        """
    if len(oldparents) == 0:
        return NULL_REVISION
    if len(oldparents) == 1:
        return oldparents[0]
    if len(newparents) == 1:
        return oldparents[1]
    try:
        return self.graph.find_unique_lca(*[oldparents[0], newparents[1]])
    except NoCommonAncestor:
        return oldparents[0]