import os
import sys
from .. import commit, config, errors, msgeditor, osutils, trace
from .. import transport as _mod_transport
from ..msgeditor import (edit_commit_message_encoded,
from ..trace import mutter
from . import (TestCaseInTempDir, TestCaseWithTransport, TestNotApplicable,
from .EncodingAdapter import encoding_scenarios
import sys
def test_commit_template_pending_merges(self):
    """Test building a commit message template when there are pending
        merges.  The commit message should show all pending merge revisions,
        as does 'status -v', not only the merge tips.
        """
    working_tree = self.make_multiple_pending_tree()
    template = msgeditor.make_commit_message_template(working_tree, None)
    self.assertEqualDiff(template, 'pending merges:\n  Bilbo Baggins 2009-01-29 Feature X finished.\n    Bilbo Baggins 2009-01-28 Feature X work.\n  Bilbo Baggins 2009-01-30 Feature Y, based on initial X work.\n')