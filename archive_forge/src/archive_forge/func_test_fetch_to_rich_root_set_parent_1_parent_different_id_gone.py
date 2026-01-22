from breezy import controldir, errors, gpg, repository
from breezy.bzr import remote
from breezy.bzr.inventory import ROOT_ID
from breezy.tests import TestNotApplicable, TestSkipped
from breezy.tests.per_repository import TestCaseWithRepository
def test_fetch_to_rich_root_set_parent_1_parent_different_id_gone(self):
    self.do_test_fetch_to_rich_root_sets_parents_correctly((), [(b'base', None, [('add', ('', ROOT_ID, 'directory', ''))]), (b'tip', None, [('unversion', ''), ('add', ('', b'my-root', 'directory', ''))])], root_id=b'my-root')