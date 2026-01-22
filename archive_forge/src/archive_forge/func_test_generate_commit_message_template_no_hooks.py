import os
import sys
from .. import commit, config, errors, msgeditor, osutils, trace
from .. import transport as _mod_transport
from ..msgeditor import (edit_commit_message_encoded,
from ..trace import mutter
from . import (TestCaseInTempDir, TestCaseWithTransport, TestNotApplicable,
from .EncodingAdapter import encoding_scenarios
import sys
def test_generate_commit_message_template_no_hooks(self):
    commit_obj = commit.Commit()
    self.assertIs(None, msgeditor.generate_commit_message_template(commit_obj))