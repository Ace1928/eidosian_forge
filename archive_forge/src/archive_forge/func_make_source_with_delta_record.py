import sys
from breezy import branch, controldir, errors, memorytree, tests
from breezy.bzr import branch as bzrbranch
from breezy.bzr import remote, versionedfile
from breezy.bzr.tests.per_repository_vf import (
from breezy.tests.scenarios import load_tests_apply_scenarios
def make_source_with_delta_record(self):
    source_repo = self.make_write_locked_repo('source')
    source_repo.start_write_group()
    key_base = (b'file-id', b'base')
    key_delta = (b'file-id', b'delta')

    def text_stream():
        yield versionedfile.FulltextContentFactory(key_base, (), None, b'lines\n')
        yield versionedfile.FulltextContentFactory(key_delta, (key_base,), None, b'more\nlines\n')
    source_repo.texts.insert_record_stream(text_stream())
    source_repo.commit_write_group()
    return source_repo