import sys
from breezy import branch, controldir, errors, memorytree, tests
from breezy.bzr import branch as bzrbranch
from breezy.bzr import remote, versionedfile
from breezy.bzr.tests.per_repository_vf import (
from breezy.tests.scenarios import load_tests_apply_scenarios
def test_insert_stream_passes_resume_info(self):
    repo = self.make_repository('test-repo')
    if not repo._format.supports_external_lookups or isinstance(repo, remote.RemoteRepository):
        raise tests.TestNotApplicable('only valid for direct connections to resumable repos')
    call_log = []
    orig = repo.get_missing_parent_inventories

    def get_missing(check_for_missing_texts=True):
        call_log.append(check_for_missing_texts)
        return orig(check_for_missing_texts=check_for_missing_texts)
    repo.get_missing_parent_inventories = get_missing
    repo.lock_write()
    self.addCleanup(repo.unlock)
    sink = repo._get_sink()
    sink.insert_stream((), repo._format, [])
    self.assertEqual([False], call_log)
    del call_log[:]
    repo.start_write_group()
    repo.texts.insert_record_stream([versionedfile.FulltextContentFactory((b'file-id', b'rev-id'), (), None, b'lines\n')])
    tokens = repo.suspend_write_group()
    self.assertNotEqual([], tokens)
    sink.insert_stream((), repo._format, tokens)
    self.assertEqual([True], call_log)