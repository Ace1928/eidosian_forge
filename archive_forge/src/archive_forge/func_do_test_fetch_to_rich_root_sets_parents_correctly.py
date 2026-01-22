from breezy import controldir, errors, gpg, repository
from breezy.bzr import remote
from breezy.bzr.inventory import ROOT_ID
from breezy.tests import TestNotApplicable, TestSkipped
from breezy.tests.per_repository import TestCaseWithRepository
def do_test_fetch_to_rich_root_sets_parents_correctly(self, result, snapshots, root_id=ROOT_ID, allow_lefthand_ghost=False):
    """Assert that result is the parents of b'tip' after fetching snapshots.

        This helper constructs a 1.9 format source, and a test-format target
        and fetches the result of building snapshots in the source, then
        asserts that the parents of tip are result.

        :param result: A parents list for the inventories.get_parent_map call.
        :param snapshots: An iterable of snapshot parameters for
            BranchBuilder.build_snapshot.
        '"""
    repo = self.make_repository('target')
    remote_format = isinstance(repo, remote.RemoteRepository)
    if not repo._format.rich_root_data and (not remote_format):
        return
    if not repo._format.supports_full_versioned_files:
        raise TestNotApplicable('format does not support full versioned files')
    builder = self.make_branch_builder('source', format='1.9')
    builder.start_series()
    for revision_id, parent_ids, actions in snapshots:
        builder.build_snapshot(parent_ids, actions, allow_leftmost_as_ghost=allow_lefthand_ghost, revision_id=revision_id)
    builder.finish_series()
    source = builder.get_branch()
    if remote_format and (not repo._format.rich_root_data):
        repo = self.make_repository('remote-target', format='1.9-rich-root')
    repo.lock_write()
    self.addCleanup(repo.unlock)
    repo.fetch(source.repository)
    graph = repo.get_file_graph()
    self.assertEqual(result, graph.get_parent_map([(root_id, b'tip')])[root_id, b'tip'])