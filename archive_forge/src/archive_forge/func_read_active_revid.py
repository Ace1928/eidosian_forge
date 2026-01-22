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
def read_active_revid(self):
    """See `RebaseState`."""
    try:
        text = self.transport.get_bytes(REBASE_CURRENT_REVID_FILENAME).rstrip(b'\n')
        if text == NULL_REVISION:
            return None
        return text
    except NoSuchFile:
        return None