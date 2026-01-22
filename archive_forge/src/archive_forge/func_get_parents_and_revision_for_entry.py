from io import BytesIO
from ... import errors
from ... import graph as _mod_graph
from ... import osutils
from ... import revision as _mod_revision
from ...bzr import inventory
from ...bzr.inventorytree import InventoryTreeChange
def get_parents_and_revision_for_entry(self, ie):
    """Get the parents and revision for an inventory entry.

        :param ie: the inventory entry
        :return parents, revision_id where
            parents is the tuple of parent revision_ids for the per-file graph
            revision_id is the revision_id to use for this entry
        """
    if self._current_rev_id is None:
        raise AssertionError('start_new_revision() must be called before get_parents_and_revision_for_entry()')
    if ie.revision != self._current_rev_id:
        raise AssertionError('start_new_revision() registered a different revision (%s) to that in the inventory entry (%s)' % (self._current_rev_id, ie.revision))
    parent_candidate_entries = ie.parent_candidates(self._rev_parent_invs)
    head_set = self._commit_builder._heads(ie.file_id, list(parent_candidate_entries))
    heads = []
    for inv in self._rev_parent_invs:
        try:
            old_rev = inv.get_entry(ie.file_id).revision
        except errors.NoSuchId:
            pass
        else:
            if old_rev in head_set:
                rev_id = inv.get_entry(ie.file_id).revision
                heads.append(rev_id)
                head_set.remove(rev_id)
    if len(heads) == 0:
        return ((), ie.revision)
    parent_entry = parent_candidate_entries[heads[0]]
    changed = False
    if len(heads) > 1:
        changed = True
    elif parent_entry.name != ie.name or parent_entry.kind != ie.kind or parent_entry.parent_id != ie.parent_id:
        changed = True
    elif ie.kind == 'file':
        if parent_entry.text_sha1 != ie.text_sha1 or parent_entry.executable != ie.executable:
            changed = True
    elif ie.kind == 'symlink':
        if parent_entry.symlink_target != ie.symlink_target:
            changed = True
    if changed:
        rev_id = ie.revision
    else:
        rev_id = parent_entry.revision
    return (tuple(heads), rev_id)