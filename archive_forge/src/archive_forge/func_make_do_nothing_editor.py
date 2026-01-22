import os
import sys
from .. import commit, config, errors, msgeditor, osutils, trace
from .. import transport as _mod_transport
from ..msgeditor import (edit_commit_message_encoded,
from ..trace import mutter
from . import (TestCaseInTempDir, TestCaseWithTransport, TestNotApplicable,
from .EncodingAdapter import encoding_scenarios
import sys
def make_do_nothing_editor(self, basename='fed'):
    if sys.platform == 'win32':
        name = basename + '.bat'
        with open(name, 'w') as f:
            f.write('@rem dummy fed')
        return name
    else:
        name = basename + '.sh'
        with open(name, 'wb') as f:
            f.write(b'#!/bin/sh\n')
        os.chmod(name, 493)
        return './' + name