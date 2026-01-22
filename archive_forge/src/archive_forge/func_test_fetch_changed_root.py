from .. import errors, osutils
from .. import revision as _mod_revision
from ..branch import Branch
from ..bzr import bzrdir, knitrepo, versionedfile
from ..upgrade import Convert
from ..workingtree import WorkingTree
from . import TestCaseWithTransport
from .test_revision import make_branches
def test_fetch_changed_root(self):
    self.make_two_commits(change_root=True, fetch_twice=False)
    self.assertEqual((), self.get_parents(b'unique-id', b'second-id'))