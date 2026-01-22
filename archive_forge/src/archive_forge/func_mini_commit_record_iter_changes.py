import os
from breezy import config, errors, osutils, repository
from breezy import revision as _mod_revision
from breezy import tests
from breezy.bzr import inventorytree
from breezy.bzr.inventorytree import InventoryTreeChange
from breezy.tests import features, per_repository
from ..test_bedding import override_whoami
def mini_commit_record_iter_changes(self, tree, name, new_name, records_version=True, delta_against_basis=True, expect_fs_hash=False):
    """Perform a miniature commit looking for record entry results.

        This version uses the record_iter_changes interface.

        :param tree: The tree to commit.
        :param name: The path in the basis tree of the tree being committed.
        :param new_name: The path in the tree being committed.
        :param records_version: True if the commit of new_name is expected to
            record a new version.
        :param delta_against_basis: True of the commit of new_name is expected
            to have a delta against the basis.
        :param expect_fs_hash: If true, looks for a fs hash output from
            record_iter_changes.
        """
    with tree.lock_write():
        parent_ids = tree.get_parent_ids()
        builder = tree.branch.get_commit_builder(parent_ids)
        try:
            parent_tree = tree.basis_tree()
            with parent_tree.lock_read():
                changes = list(tree.iter_changes(parent_tree))
            result = list(builder.record_iter_changes(tree, parent_ids[0], changes))
            self.assertTrue(tree.is_versioned(new_name))
            if isinstance(tree, inventorytree.InventoryTree):
                file_id = tree.path2id(new_name)
                self.assertIsNot(None, file_id)
                if expect_fs_hash:
                    tree_file_stat = tree.get_file_with_stat(new_name)
                    tree_file_stat[0].close()
                    self.assertLength(1, result)
                    result = result[0]
                    self.assertEqual(result[0], new_name)
                    self.assertEqual(result[1][0], tree.get_file_sha1(new_name))
                    self.assertEqualStat(result[1][1], tree_file_stat[1])
                else:
                    self.assertEqual([], result)
            builder.finish_inventory()
            if tree.branch.repository._format.supports_full_versioned_files:
                inv_key = (builder._new_revision_id,)
                inv_sha1 = tree.branch.repository.inventories.get_sha1s([inv_key])[inv_key]
                self.assertEqual(inv_sha1, builder.inv_sha1)
            rev2 = builder.commit('rev2')
        except BaseException:
            builder.abort()
            raise
        delta = builder.get_basis_delta()
        delta_dict = {change[1]: change for change in delta}
        if tree.branch.repository._format.records_per_file_revision:
            version_recorded = new_name in delta_dict and delta_dict[new_name][3] is not None and (delta_dict[new_name][3].revision == rev2)
            if records_version:
                self.assertTrue(version_recorded)
            else:
                self.assertFalse(version_recorded)
        revtree = builder.revision_tree()
        new_entry = next(revtree.iter_entries_by_dir(specific_files=[new_name]))[1]
        if delta_against_basis:
            delta_old_name, delta_new_name, delta_file_id, delta_entry = delta_dict[new_name]
            self.assertEqual(delta_new_name, new_name)
            if tree.supports_rename_tracking():
                self.assertEqual(name, delta_old_name)
            else:
                self.assertIn(delta_old_name, (name, None))
            if tree.supports_setting_file_ids():
                self.assertEqual(delta_file_id, file_id)
                self.assertEqual(delta_entry.file_id, file_id)
            self.assertEqual(delta_entry.kind, new_entry.kind)
            self.assertEqual(delta_entry.name, new_entry.name)
            self.assertEqual(delta_entry.parent_id, new_entry.parent_id)
            if delta_entry.kind == 'file':
                self.assertEqual(delta_entry.text_size, revtree.get_file_size(new_name))
                if getattr(delta_entry, 'text_sha1', None):
                    self.assertEqual(delta_entry.text_sha1, revtree.get_file_sha1(new_name))
            elif delta_entry.kind == 'symlink':
                self.assertEqual(delta_entry.symlink_target, new_entry.symlink_target)
        else:
            expected_delta = None
            if tree.branch.repository._format.records_per_file_revision:
                self.assertFalse(version_recorded)
        tree.set_parent_ids([rev2])
    return rev2