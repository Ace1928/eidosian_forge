import doctest
import os
import re
import sys
from testtools.matchers import DocTestMatches
from ... import config, ignores, msgeditor, osutils
from ...controldir import ControlDir
from .. import TestCaseWithTransport, features, test_foreign
from ..test_bedding import override_whoami
def setup_editor(self):
    if sys.platform == 'win32':
        with open('fed.bat', 'w') as f:
            f.write('@rem dummy fed')
        self.overrideEnv('BRZ_EDITOR', 'fed.bat')
    else:
        with open('fed.sh', 'wb') as f:
            f.write(b'#!/bin/sh\n')
        os.chmod('fed.sh', 493)
        self.overrideEnv('BRZ_EDITOR', './fed.sh')