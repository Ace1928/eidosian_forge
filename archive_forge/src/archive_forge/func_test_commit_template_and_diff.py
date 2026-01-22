import os
import sys
from .. import commit, config, errors, msgeditor, osutils, trace
from .. import transport as _mod_transport
from ..msgeditor import (edit_commit_message_encoded,
from ..trace import mutter
from . import (TestCaseInTempDir, TestCaseWithTransport, TestNotApplicable,
from .EncodingAdapter import encoding_scenarios
import sys
def test_commit_template_and_diff(self):
    """Test building a commit message template"""
    working_tree = self.make_uncommitted_tree()
    template = make_commit_message_template_encoded(working_tree, None, diff=True, output_encoding='utf8')
    self.assertTrue(b'@@ -0,0 +1,1 @@\n+contents of hello\n' in template)
    self.assertTrue('added:\n  hellØ\n'.encode() in template)