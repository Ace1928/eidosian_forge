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
def regenerate_default_revid(repository, revid):
    """Generate a revision id for the rebase of an existing revision.

    :param repository: Repository in which the revision is present.
    :param revid: Revision id of the revision that is being rebased.
    :return: new revision id."""
    if revid == NULL_REVISION:
        return NULL_REVISION
    rev = repository.get_revision(revid)
    return gen_revision_id(rev.committer, rev.timestamp)