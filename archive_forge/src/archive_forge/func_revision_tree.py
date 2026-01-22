import base64
import os
import pprint
from io import BytesIO
from ... import cache_utf8, osutils, timestamp
from ...errors import BzrError, NoSuchId, TestamentMismatch
from ...osutils import pathjoin, sha_string, sha_strings
from ...revision import NULL_REVISION, Revision
from ...trace import mutter, warning
from ...tree import InterTree, Tree
from ..inventory import (Inventory, InventoryDirectory, InventoryFile,
from ..inventorytree import InventoryTree
from ..testament import StrictTestament
from ..xml5 import serializer_v5
from . import apply_bundle
def revision_tree(self, repository, revision_id, base=None):
    revision = self.get_revision(revision_id)
    base = self.get_base(revision)
    if base == revision_id:
        raise AssertionError()
    if not self._validated_revisions_against_repo:
        self._validate_references_from_repository(repository)
    revision_info = self.get_revision_info(revision_id)
    inventory_revision_id = revision_id
    bundle_tree = BundleTree(repository.revision_tree(base), inventory_revision_id)
    self._update_tree(bundle_tree, revision_id)
    inv = bundle_tree.inventory
    self._validate_inventory(inv, revision_id)
    self._validate_revision(bundle_tree, revision_id)
    return bundle_tree