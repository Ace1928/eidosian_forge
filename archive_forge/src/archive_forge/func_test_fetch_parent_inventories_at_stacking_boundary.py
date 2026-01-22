import sys
from breezy import errors, osutils, repository
from breezy.bzr import inventory, versionedfile
from breezy.bzr.vf_search import SearchResult
from breezy.errors import NoSuchRevision
from breezy.repository import WriteGroup
from breezy.revision import NULL_REVISION, Revision
from breezy.tests import TestNotApplicable
from breezy.tests.per_interrepository import TestCaseWithInterRepository
from breezy.tests.per_interrepository.test_interrepository import \
def test_fetch_parent_inventories_at_stacking_boundary(self):
    """Fetch to a stacked branch copies inventories for parents of
        revisions at the stacking boundary.

        This is necessary so that the server is able to determine the file-ids
        altered by all revisions it contains, which means that it needs both
        the inventory for any revision it has, and the inventories of all that
        revision's parents.

        However, we should also skip any revisions which are ghosts in the
        parents.
        """
    if not self.repository_format_to.supports_external_lookups:
        raise TestNotApplicable('Need stacking support in the target.')
    builder = self.make_branch_builder('branch')
    builder.start_series()
    base = builder.build_snapshot(None, [('add', ('', None, 'directory', '')), ('add', ('file', None, 'file', b'content\n'))])
    left = builder.build_snapshot([base], [('modify', ('file', b'left content\n'))])
    right = builder.build_snapshot([base], [('modify', ('file', b'right content\n'))])
    merge = builder.build_snapshot([left, right], [('modify', ('file', b'left and right content\n'))])
    builder.finish_series()
    branch = builder.get_branch()
    revtree = branch.repository.revision_tree(merge)
    if not revtree.supports_file_ids:
        raise TestNotApplicable('from format does not support file ids')
    root_id = revtree.path2id('')
    file_id = revtree.path2id('file')
    repo = self.make_to_repository('trunk')
    trunk = repo.controldir.create_branch()
    trunk.repository.fetch(branch.repository, left)
    trunk.repository.fetch(branch.repository, right)
    repo = self.make_to_repository('stacked')
    stacked_branch = repo.controldir.create_branch()
    stacked_branch.set_stacked_on_url(trunk.base)
    stacked_branch.repository.fetch(branch.repository, merge)
    unstacked_repo = stacked_branch.controldir.open_repository()
    unstacked_repo.lock_read()
    self.addCleanup(unstacked_repo.unlock)
    self.assertFalse(unstacked_repo.has_revision(left))
    self.assertFalse(unstacked_repo.has_revision(right))
    self.assertEqual({(left,), (right,), (merge,)}, unstacked_repo.inventories.keys())
    trunk.lock_read()
    self.addCleanup(trunk.unlock)
    left_tree, right_tree = trunk.repository.revision_trees([left, right])
    stacked_branch.lock_read()
    self.addCleanup(stacked_branch.unlock)
    stacked_left_tree, stacked_right_tree = stacked_branch.repository.revision_trees([left, right])
    self.assertEqual(left_tree.root_inventory, stacked_left_tree.root_inventory)
    self.assertEqual(right_tree.root_inventory, stacked_right_tree.root_inventory)
    self.assertTrue(unstacked_repo.has_revision(merge))
    expected_texts = {(file_id, merge)}
    if stacked_branch.repository.texts.get_parent_map([(root_id, merge)]):
        expected_texts.add((root_id, merge))
    self.assertEqual(expected_texts, unstacked_repo.texts.keys())
    self.assertCanStreamRevision(unstacked_repo, merge)