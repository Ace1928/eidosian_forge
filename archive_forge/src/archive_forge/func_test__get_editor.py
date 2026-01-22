import os
import sys
from .. import commit, config, errors, msgeditor, osutils, trace
from .. import transport as _mod_transport
from ..msgeditor import (edit_commit_message_encoded,
from ..trace import mutter
from . import (TestCaseInTempDir, TestCaseWithTransport, TestNotApplicable,
from .EncodingAdapter import encoding_scenarios
import sys
def test__get_editor(self):
    self.overrideEnv('BRZ_EDITOR', 'bzr_editor')
    self.overrideEnv('VISUAL', 'visual')
    self.overrideEnv('EDITOR', 'editor')
    conf = config.GlobalStack()
    conf.store._load_from_string(b'[DEFAULT]\neditor = config_editor\n')
    conf.store.save()
    editors = list(msgeditor._get_editor())
    editors = [editor for editor, cfg_src in editors]
    self.assertEqual(['bzr_editor', 'config_editor', 'visual', 'editor'], editors[:4])
    if sys.platform == 'win32':
        self.assertEqual(['wordpad.exe', 'notepad.exe'], editors[4:])
    else:
        self.assertEqual(['/usr/bin/editor', 'vi', 'pico', 'nano', 'joe'], editors[4:])