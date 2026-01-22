import os
import sys
from .. import commit, config, errors, msgeditor, osutils, trace
from .. import transport as _mod_transport
from ..msgeditor import (edit_commit_message_encoded,
from ..trace import mutter
from . import (TestCaseInTempDir, TestCaseWithTransport, TestNotApplicable,
from .EncodingAdapter import encoding_scenarios
import sys
def test_edit_commit_message_without_infotext(self):
    self.make_uncommitted_tree()
    self.make_fake_editor()
    mutter('edit_commit_message without infotext')
    self.assertEqual('test message from fed\n', msgeditor.edit_commit_message(''))