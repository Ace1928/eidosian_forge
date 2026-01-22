from io import BytesIO
from ... import errors
from ... import graph as _mod_graph
from ... import osutils
from ... import revision as _mod_revision
from ...bzr import inventory
from ...bzr.inventorytree import InventoryTreeChange
def load_using_delta(self, rev, basis_inv, inv_delta, signature, text_provider, parents_provider, inventories_provider=None):
    """Load a revision by applying a delta to a (CHK)Inventory.

        :param rev: the Revision
        :param basis_inv: the basis Inventory or CHKInventory
        :param inv_delta: the inventory delta
        :param signature: signing information
        :param text_provider: a callable expecting a file_id parameter
            that returns the text for that file-id
        :param parents_provider: a callable expecting a file_id parameter
            that return the list of parent-ids for that file-id
        :param inventories_provider: a callable expecting a repository and
            a list of revision-ids, that returns:
              * the list of revision-ids present in the repository
              * the list of inventories for the revision-id's,
                including an empty inventory for the missing revisions
            If None, a default implementation is provided.
        """
    builder = self.repo._commit_builder_class(self.repo, parents=rev.parent_ids, config=None, timestamp=rev.timestamp, timezone=rev.timezone, committer=rev.committer, revprops=rev.properties, revision_id=rev.revision_id)
    if self._graph is None and self._use_known_graph:
        if getattr(_mod_graph, 'GraphThunkIdsToKeys', None) and getattr(_mod_graph.GraphThunkIdsToKeys, 'add_node', None) and getattr(self.repo, 'get_known_graph_ancestry', None):
            self._graph = self.repo.get_known_graph_ancestry(rev.parent_ids)
        else:
            self._use_known_graph = False
    if self._graph is not None:
        orig_heads = builder._heads

        def thunked_heads(file_id, revision_ids):
            if len(revision_ids) < 2:
                res = set(revision_ids)
            else:
                res = set(self._graph.heads(revision_ids))
            return res
        builder._heads = thunked_heads
    if rev.parent_ids:
        basis_rev_id = rev.parent_ids[0]
    else:
        basis_rev_id = _mod_revision.NULL_REVISION
    tree = _TreeShim(self.repo, basis_inv, inv_delta, text_provider)
    changes = tree._delta_to_iter_changes()
    for path, fs_hash in builder.record_iter_changes(tree, basis_rev_id, changes):
        pass
    builder.finish_inventory()
    if isinstance(builder.inv_sha1, tuple):
        builder.inv_sha1, builder.new_inventory = builder.inv_sha1
    rev.inv_sha1 = builder.inv_sha1
    config = builder._config_stack
    builder.repository.add_revision(builder._new_revision_id, rev, builder.revision_tree().root_inventory)
    if self._graph is not None:
        self._graph.add_node(builder._new_revision_id, rev.parent_ids)
    if signature is not None:
        raise AssertionError('signatures not guaranteed yet')
        self.repo.add_signature_text(rev.revision_id, signature)
    return builder.revision_tree().root_inventory