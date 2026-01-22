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
def commit_rebase(self, oldrev, newrevid):
    """Commit a rebase.

        :param oldrev: Revision info of new revision to commit.
        :param newrevid: New revision id."""
    assert oldrev.revision_id != newrevid, 'Invalid revid %r' % newrevid
    revprops = dict(oldrev.properties)
    revprops[REVPROP_REBASE_OF] = oldrev.revision_id.decode('utf-8')
    committer = self.wt.branch.get_config().username()
    authors = oldrev.get_apparent_authors()
    if oldrev.committer == committer:
        if [oldrev.committer] == authors:
            authors = None
    elif oldrev.committer not in authors:
        authors.append(oldrev.committer)
    if 'author' in revprops:
        del revprops['author']
    if 'authors' in revprops:
        del revprops['authors']
    self.wt.commit(message=oldrev.message, timestamp=oldrev.timestamp, timezone=oldrev.timezone, revprops=revprops, rev_id=newrevid, committer=committer, authors=authors)