from io import StringIO
from .. import config
from .. import status as _mod_status
from ..revisionspec import RevisionSpec
from ..status import show_pending_merges, show_tree_status
from . import TestCaseWithTransport
def test_multiple_pending_verbose(self):
    tree = self.make_multiple_pending_tree()
    output = StringIO()
    tree.lock_read()
    self.addCleanup(tree.unlock)
    show_pending_merges(tree, output, verbose=True)
    self.assertEqualDiff('pending merges:\n  Joe Foo 2007-12-04 commit 3b\n    Joe Foo 2007-12-04 commit 2b\n  Joe Foo 2007-12-04 commit 3c\n', output.getvalue())