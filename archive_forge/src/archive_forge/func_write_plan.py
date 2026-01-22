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
def write_plan(self, replace_map):
    """See `RebaseState`."""
    self.wt.update_feature_flags({b'rebase-v1': b'write-required'})
    content = marshall_rebase_plan(self.wt.branch.last_revision_info(), replace_map)
    assert isinstance(content, bytes)
    self.transport.put_bytes(REBASE_PLAN_FILENAME, content)