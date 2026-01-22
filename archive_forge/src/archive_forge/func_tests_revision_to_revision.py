from io import StringIO
from .. import config
from .. import status as _mod_status
from ..revisionspec import RevisionSpec
from ..status import show_pending_merges, show_tree_status
from . import TestCaseWithTransport
def tests_revision_to_revision(self):
    """doing a status between two revision trees should work."""
    tree = self.make_branch_and_tree('.')
    r1_id = tree.commit('one', allow_pointless=True)
    r2_id = tree.commit('two', allow_pointless=True)
    output = StringIO()
    show_tree_status(tree, to_file=output, revision=[RevisionSpec.from_string('revid:%s' % r1_id.decode('utf-8')), RevisionSpec.from_string('revid:%s' % r2_id.decode('utf-8'))])