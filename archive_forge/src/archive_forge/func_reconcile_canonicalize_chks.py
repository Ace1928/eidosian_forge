import time
from .. import controldir, debug, errors, osutils
from .. import revision as _mod_revision
from .. import trace, ui
from ..bzr import chk_map, chk_serializer
from ..bzr import index as _mod_index
from ..bzr import inventory, pack, versionedfile
from ..bzr.btree_index import BTreeBuilder, BTreeGraphIndex
from ..bzr.groupcompress import GroupCompressVersionedFiles, _GCGraphIndex
from ..bzr.vf_repository import StreamSource
from .pack_repo import (NewPack, Pack, PackCommitBuilder, Packer,
from .static_tuple import StaticTuple
def reconcile_canonicalize_chks(self):
    """Reconcile this repository to make sure all CHKs are in canonical
        form.
        """
    from .reconcile import PackReconciler
    with self.lock_write():
        reconciler = PackReconciler(self, thorough=True, canonicalize_chks=True)
        return reconciler.reconcile()