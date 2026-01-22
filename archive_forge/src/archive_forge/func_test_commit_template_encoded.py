import os
import sys
from .. import commit, config, errors, msgeditor, osutils, trace
from .. import transport as _mod_transport
from ..msgeditor import (edit_commit_message_encoded,
from ..trace import mutter
from . import (TestCaseInTempDir, TestCaseWithTransport, TestNotApplicable,
from .EncodingAdapter import encoding_scenarios
import sys
def test_commit_template_encoded(self):
    """Test building a commit message template"""
    working_tree = self.make_uncommitted_tree()
    template = make_commit_message_template_encoded(working_tree, None, output_encoding='utf8')
    self.assertEqualDiff(template, 'added:\n  hellØ\n'.encode())